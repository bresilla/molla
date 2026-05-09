//! WGSL compute kernels and their Rust dispatch wrappers.

#[cfg(feature = "gpu")]
mod actuator_delay;
#[cfg(feature = "gpu")]
mod actuator_pd;
#[cfg(feature = "gpu")]
mod actuator_pid;
#[cfg(feature = "gpu")]
mod actuator_neural_mlp;
#[cfg(feature = "gpu")]
mod actuator_pipeline_pd;
#[cfg(feature = "gpu")]
mod actuator_pipeline_pid;
#[cfg(feature = "gpu")]
mod clamping_dc_motor;
#[cfg(feature = "gpu")]
mod clamping_max_effort;
#[cfg(feature = "gpu")]
mod clamping_position_based;
#[cfg(feature = "gpu")]
mod contact_sensor;
#[cfg(feature = "gpu")]
mod crba;
#[cfg(feature = "gpu")]
mod dense3;
#[cfg(feature = "gpu")]
mod epa;
#[cfg(feature = "gpu")]
mod featherstone;
// Adjoint dispatcher is feature-gated separately — it does not (yet)
// require GPU and is needed by the autograd CPU path + viewer
// regardless of the `gpu` feature.
mod featherstone_adjoint;
#[cfg(feature = "gpu")]
mod fft;
#[cfg(feature = "gpu")]
mod gjk;
#[cfg(feature = "gpu")]
mod mesh_bvh_raycast;
#[cfg(feature = "gpu")]
mod frame_transform;
#[cfg(feature = "gpu")]
mod imu;
#[cfg(feature = "gpu")]
mod integrate_bodies;
#[cfg(feature = "gpu")]
mod jacobian;
#[cfg(feature = "gpu")]
mod lbvh;
#[cfg(feature = "gpu")]
mod radix_sort;
#[cfg(feature = "gpu")]
mod raycast;
#[cfg(feature = "gpu")]
mod robust_loss;
#[cfg(feature = "gpu")]
mod rnea;
#[cfg(feature = "gpu")]
mod sparse_grid;
#[cfg(feature = "gpu")]
mod tiled_camera;
#[cfg(feature = "gpu")]
mod xpbd;
#[cfg(feature = "gpu")]
mod xpbd_equality;
#[cfg(feature = "gpu")]
mod xpbd_particles;
#[cfg(feature = "gpu")]
mod vbd_cloth;
#[cfg(feature = "gpu")]
mod dahl_friction;

#[cfg(feature = "gpu")]
pub use actuator_delay::ActuatorDelayKernel;
#[cfg(feature = "gpu")]
pub use actuator_pd::ActuatorPdKernel;
#[cfg(feature = "gpu")]
pub use actuator_pid::ActuatorPidKernel;
#[cfg(feature = "gpu")]
pub use actuator_neural_mlp::ActuatorNeuralMlpKernel;
#[cfg(feature = "gpu")]
pub use actuator_pipeline_pd::ActuatorPipelinePdKernel;
#[cfg(feature = "gpu")]
pub use actuator_pipeline_pid::ActuatorPipelinePidKernel;
#[cfg(feature = "gpu")]
pub use clamping_dc_motor::ClampingDcMotorKernel;
#[cfg(feature = "gpu")]
pub use clamping_max_effort::ClampingMaxEffortKernel;
#[cfg(feature = "gpu")]
pub use clamping_position_based::ClampingPositionBasedKernel;
#[cfg(feature = "gpu")]
pub use contact_sensor::ContactAggregateKernel;
#[cfg(feature = "gpu")]
pub use crba::MassMatrixKernel;
#[cfg(feature = "gpu")]
pub use dense3::{Dense3Kernel, Dense3Op, Dense3Result};
#[cfg(feature = "gpu")]
pub use epa::{
    decode_epa_out, EpaKernel, EpaPairResult, EPA_OUT_STRIDE, EPA_STATUS_FALLBACK, EPA_STATUS_NONE,
    EPA_STATUS_OK,
};
#[cfg(feature = "gpu")]
pub use featherstone::FeatherstoneKernel;
pub use featherstone_adjoint::{
    time_forward_us, AdjointBackend, AdjointResult, FeatherstoneAdjointDispatcher,
};
#[cfg(feature = "gpu")]
pub use fft::{Complex32, FftKernel};
#[cfg(feature = "gpu")]
pub use gjk::{GjkKernel, GJK_SIMPLEX_STRIDE};
#[cfg(feature = "gpu")]
pub use mesh_bvh_raycast::{MeshBvhGpu, MeshBvhRaycastKernel};
#[cfg(feature = "gpu")]
pub use frame_transform::FrameTransformKernel;
#[cfg(feature = "gpu")]
pub use imu::ImuKernel;
#[cfg(feature = "gpu")]
pub use integrate_bodies::IntegrateBodiesKernel;
#[cfg(feature = "gpu")]
pub use jacobian::{jacobian_gpu, JacobianKernel, JacobianRequest};
#[cfg(feature = "gpu")]
pub use lbvh::{collect_shape_aabbs, LbvhBroadPhase, LbvhPairs};
#[cfg(feature = "gpu")]
pub use radix_sort::{radix_sort_pairs, RadixSortKernel};
#[cfg(feature = "gpu")]
pub use raycast::RaycastKernel;
#[cfg(feature = "gpu")]
pub use robust_loss::{LossPair, RobustLossKernel};
#[cfg(feature = "gpu")]
pub use rnea::{inverse_dynamics_gpu, InverseDynamicsKernel};
#[cfg(feature = "gpu")]
pub use sparse_grid::SparseGridKernel;
#[cfg(feature = "gpu")]
pub use tiled_camera::{CameraRaysKernel, CameraTraceKernel};
#[cfg(feature = "gpu")]
pub use xpbd::XpbdKernel;
#[cfg(feature = "gpu")]
pub use xpbd_equality::XpbdEqualityKernel;
#[cfg(feature = "gpu")]
pub use xpbd_particles::XpbdParticlesKernel;
#[cfg(feature = "gpu")]
pub use vbd_cloth::VbdClothKernel;
#[cfg(feature = "gpu")]
pub use dahl_friction::{dahl_oracle_step, DahlConfig, DahlFrictionKernel};
