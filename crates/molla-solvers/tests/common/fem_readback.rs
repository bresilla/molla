use molla_compute::Engine;
use molla_math::{Quat, Real, Transform, Vec3};
use molla_solvers::fem_rigid_gpu::FemRigidGpuSystem;

pub fn body_poses(engine: &Engine, system: &FemRigidGpuSystem, count: usize) -> Vec<Transform> {
    super::gpu_readback::read::<[f32; 8]>(
        engine,
        system.rigid_state().body_q.device_buffer().unwrap(),
        count,
    )
    .iter()
    .map(|p| Transform {
        position: Vec3::new(p[0] as Real, p[1] as Real, p[2] as Real),
        rotation: Quat::from_xyzw(p[4] as Real, p[5] as Real, p[6] as Real, p[7] as Real)
            .normalize(),
    })
    .collect()
}
