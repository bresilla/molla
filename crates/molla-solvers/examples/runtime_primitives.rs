//! Public live-runtime distance, cone, reaction, break, friction and D6 examples.

use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxes, JointAxis, JointMotor};
use molla_sim::runtime::{
    BodyDesc, BodyHandle, ColliderDesc, ColliderGeometry, DistanceLimits, JointBreakLimits,
    JointDesc, JointHandle, RigidScene, SphericalConeLimits,
};
use molla_sim::{
    BodyParams, ColliderMass, ContactMaterial, FrictionCoefficients, JointTargetMode, JointType,
    MassProperties,
};
use molla_solvers::rigid_world::RigidWorld;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const CASES: &[&str] = &["distance", "cone", "reaction", "break", "friction", "d6"];

fn scene() -> Result<RigidScene> {
    let mut scene = RigidScene::new()?;
    scene.set_gravity(-Vec3::Y * 10.)?;
    Ok(scene)
}

fn body(scene: &mut RigidScene, position: Vec3, local_com: Vec3) -> Result<BodyHandle> {
    Ok(scene.insert_body(BodyDesc {
        params: BodyParams::new().with_position(position),
        additional_mass: Some(MassProperties {
            mass: 2.,
            local_com,
            inertia: Mat3::IDENTITY,
        }),
        ..Default::default()
    })?)
}

fn world_joint(scene: &mut RigidScene, child: BodyHandle, kind: JointType) -> Result<JointHandle> {
    Ok(scene.insert_joint(JointDesc {
        parent: None,
        child,
        kind,
        axis: Vec3::X,
        frame_parent: Transform::IDENTITY,
        frame_child: Transform::IDENTITY,
    })?)
}

enum Execution {
    Cpu,
    #[cfg(feature = "gpu")]
    Gpu(std::sync::Arc<molla_compute::Engine>),
}

impl Execution {
    fn new(gpu: bool) -> Result<Self> {
        if !gpu {
            return Ok(Self::Cpu);
        }
        #[cfg(feature = "gpu")]
        {
            let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu()?);
            let adapter = engine
                .adapter_info()
                .ok_or("missing GPU adapter information")?;
            if adapter.is_software {
                return Err("primitive GPU examples require a hardware adapter".into());
            }
            println!("adapter={adapter:?}");
            Ok(Self::Gpu(engine))
        }
        #[cfg(not(feature = "gpu"))]
        {
            Err("GPU selection requires the gpu feature; no CPU fallback".into())
        }
    }
}

fn world(scene: RigidScene, execution: &Execution) -> Result<RigidWorld> {
    let mut world = match execution {
        Execution::Cpu => RigidWorld::from_scene(scene)?,
        #[cfg(feature = "gpu")]
        Execution::Gpu(engine) => RigidWorld::from_scene_gpu(scene, engine.clone())?,
    };
    world.solver.substeps = 4;
    Ok(world)
}

fn advance(world: &mut RigidWorld) -> Result<()> {
    for _ in 0..240 {
        world.step(1. / 240.)?;
    }
    Ok(())
}

fn distance(execution: &Execution) -> Result<()> {
    let mut scene = scene()?;
    let body = body(&mut scene, -Vec3::Y, Vec3::ZERO)?;
    let joint = world_joint(&mut scene, body, JointType::Distance)?;
    scene.set_distance_limits(
        joint,
        DistanceLimits {
            min: Some(1.),
            max: Some(1.),
        },
    )?;
    let mut world = world(scene, execution)?;
    advance(&mut world)?;
    println!(
        "distance={:?}",
        world
            .distance_reading(joint)
            .ok_or("missing distance sample")?
    );
    Ok(())
}

fn cone(execution: &Execution) -> Result<()> {
    let mut scene = scene()?;
    let body = body(&mut scene, Vec3::ZERO, Vec3::X * 0.2)?;
    let joint = world_joint(&mut scene, body, JointType::Ball)?;
    scene.set_cone_limits(
        joint,
        SphericalConeLimits {
            angle0: Some(0.25),
            angle1: Some(0.4),
        },
    )?;
    let mut world = world(scene, execution)?;
    advance(&mut world)?;
    println!(
        "cone={:?}",
        world.cone_reading(joint).ok_or("missing cone sample")?
    );
    Ok(())
}

fn reaction(execution: &Execution) -> Result<()> {
    let mut scene = scene()?;
    let body = body(&mut scene, Vec3::ZERO, Vec3::ZERO)?;
    let joint = world_joint(&mut scene, body, JointType::Fixed)?;
    let mut world = world(scene, execution)?;
    world.set_reaction_readout(true)?;
    world.step(1. / 240.)?;
    println!(
        "reaction={:?}",
        world
            .joint_reaction(joint)
            .ok_or("missing reaction sample")?
    );
    Ok(())
}

fn breaking(execution: &Execution) -> Result<()> {
    let mut scene = scene()?;
    let body = body(&mut scene, Vec3::ZERO, Vec3::ZERO)?;
    let joint = world_joint(&mut scene, body, JointType::Fixed)?;
    scene.set_joint_break_limits(
        joint,
        JointBreakLimits {
            force: Some(5.),
            torque: None,
        },
    )?;
    let mut world = world(scene, execution)?;
    world.step(1. / 240.)?;
    let events = world.drain_joint_break_events();
    if events.len() != 1 || events[0].joint != joint {
        return Err("expected exactly one event for the authored breakable joint".into());
    }
    println!("break={:?}", events[0]);
    world.step(1. / 240.)?;
    if !world.joint_break_events().is_empty() {
        return Err("detached joint emitted a repeated break event".into());
    }
    println!(
        "detached_body_velocity={:?}",
        world.scene.body_velocity(body)?
    );
    Ok(())
}

fn friction(execution: &Execution) -> Result<()> {
    let mut scene = scene()?;
    let body = body(&mut scene, Vec3::Y * 0.5, Vec3::ZERO)?;
    let material = ContactMaterial {
        stiffness: 1e5,
        damping: 300.,
        friction: Some(FrictionCoefficients {
            mu_static: 0.8,
            mu_dynamic: 0.4,
        }),
        ..Default::default()
    };
    let mut ball = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    ball.parent = Some(body);
    ball.mass = ColliderMass::Mass(0.);
    ball.contact_material = material;
    scene.insert_collider(ball)?;
    let mut floor = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(10., 0.1, 10.),
    });
    floor.pose = Transform::from_translation(-Vec3::Y * 0.1);
    floor.contact_material = material;
    scene.insert_collider(floor)?;
    scene.set_body_velocity(body, SpatialVector::new(Vec3::X, Vec3::ZERO))?;
    let mut world = world(scene, execution)?;
    advance(&mut world)?;
    let contacts = world.contact_pairs();
    if contacts.is_empty() {
        return Err("expected a supported ball-floor contact".into());
    }
    println!("friction_contacts={contacts:?}");
    println!(
        "rolling_body_velocity={:?}",
        world.scene.body_velocity(body)?
    );
    Ok(())
}

fn d6(execution: &Execution) -> Result<()> {
    let mut scene = scene()?;
    scene.set_gravity(Vec3::ZERO)?;
    let body = body(&mut scene, Vec3::ZERO, Vec3::ZERO)?;
    let joint = world_joint(&mut scene, body, JointType::D6)?;
    scene.set_joint_locked_axes(joint, JointAxes(JointAxes::ALL.0 & !1))?;
    scene.set_joint_motor(
        joint,
        JointAxis::LinX,
        JointMotor {
            mode: JointTargetMode::PositionVelocity,
            target_position: 0.25,
            target_velocity: 0.,
            stiffness: 30.,
            damping: 10.,
            max_force: 5.,
            ..Default::default()
        },
    )?;
    let mut world = world(scene, execution)?;
    advance(&mut world)?;
    println!("d6_pose={:?}", world.scene.body_pose(body)?);
    Ok(())
}

fn main() -> Result<()> {
    let mut gpu = false;
    let mut selected = "all".to_owned();
    let mut selected_once = false;
    for arg in std::env::args().skip(1) {
        if arg == "--gpu" {
            gpu = true;
        } else if arg == "--help" {
            println!("runtime_primitives [all|distance|cone|reaction|break|friction|d6] [--gpu]");
            return Ok(());
        } else if !selected_once && (arg == "all" || CASES.contains(&arg.as_str())) {
            selected = arg;
            selected_once = true;
        } else {
            return Err(format!("unknown or duplicate example argument: {arg}").into());
        }
    }
    let execution = Execution::new(gpu)?;
    for &case in CASES {
        if selected != "all" && selected != case {
            continue;
        }
        println!("example={case} backend={}", if gpu { "gpu" } else { "cpu" });
        match case {
            "distance" => distance(&execution)?,
            "cone" => cone(&execution)?,
            "reaction" => reaction(&execution)?,
            "break" => breaking(&execution)?,
            "friction" => friction(&execution)?,
            "d6" => d6(&execution)?,
            _ => unreachable!(),
        }
    }
    Ok(())
}
