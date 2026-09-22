use molla_core::{BodyId, WorldId};
use molla_math::{Mat3, SpatialInertia, Transform, Vec3};
use molla_sim::{BodyParams, ModelBuilder};
use molla_solvers::fem_rigid::FemRigidConfig;
use molla_solvers::fem_rigid_gpu::FemRigidGpuScene;

pub fn scene() -> FemRigidGpuScene {
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), Vec3::ZERO);
    builder.set_linear_damping(0.0);
    for point in [Vec3::ZERO, Vec3::X * 0.1, Vec3::Y * 0.1, Vec3::Z * 0.1] {
        builder.add_particle(point, 0.1, 0.0, WorldId(0));
    }
    builder
        .add_tet_mesh(&[[0, 1, 2, 3]], 2e6, 0.45, None)
        .unwrap();
    let soft_model = builder.build().unwrap();
    let soft_state = soft_model.state().unwrap();
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), Vec3::ZERO);
    builder.begin_articulation();
    let rotor = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass: 1.0,
        com: Vec3::ZERO,
        inertia: Mat3::from_diagonal(Vec3::splat(0.25)),
    }));
    builder
        .add_joint_revolute(
            BodyId::NONE,
            rotor,
            Transform::IDENTITY,
            Transform::IDENTITY,
            Vec3::X,
        )
        .unwrap();
    let mut rigid_model = builder.build().unwrap();
    rigid_model.joint_effort_limit.host_mut().unwrap()[0] = 1.0;
    let rigid_state = rigid_model.state().unwrap();
    let control = rigid_model.control();
    FemRigidGpuScene {
        soft_model,
        soft_state,
        rigid_model,
        rigid_state,
        control,
        shapes: Vec::new(),
    }
}

pub fn config() -> FemRigidConfig {
    FemRigidConfig {
        max_substep: 1.0 / 4096.0,
        elastic_iterations: 4,
        ..Default::default()
    }
}
