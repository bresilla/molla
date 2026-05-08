//! Semi-implicit Euler integrator for rigid bodies.
//!
//! CPU path is a direct port of Newton's `integrate_rigid_body` (see
//! `xtra/newton/newton/_src/solvers/solver.py`); the GPU path runs the
//! same algorithm in WGSL via [`crate::kernels::IntegrateBodiesKernel`].

use crate::core::{Error, Result};
use crate::math::{Mat3, Quat, Real, SpatialVector, Transform, Vec3};
use crate::sim::{BodyFlags, Contacts, Control, Model, State};

#[cfg(feature = "gpu")]
use std::sync::Arc;

use super::Solver;

/// Backend selector for [`SemiImplicitSolver`].
#[allow(clippy::large_enum_variant, reason = "GPU variant holds Arcs and a kernel; the size delta is fine for a per-solver enum that lives once")]
pub enum SemiImplicitBackend {
    /// Single-threaded CPU loop.
    Cpu,
    /// GPU dispatch via wgpu. The model and both states must be uploaded
    /// (see `Model::upload_to_device`, `State::upload_to_device`) before
    /// calling [`Solver::step`].
    #[cfg(feature = "gpu")]
    Gpu(GpuBackend),
}

/// GPU bookkeeping for [`SemiImplicitBackend::Gpu`]. Holds the kernel,
/// device handles, and a 1-element dummy `body_external_f` buffer used
/// when the user-supplied [`Control`] has no per-body wrenches.
#[cfg(feature = "gpu")]
pub struct GpuBackend {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: Arc<wgpu::Queue>,
    pub(crate) kernel: crate::kernels::IntegrateBodiesKernel,
    pub(crate) external_f_dummy: wgpu::Buffer,
}

#[cfg(feature = "gpu")]
impl GpuBackend {
    fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let kernel = crate::kernels::IntegrateBodiesKernel::new(&device);
        // 1-element dummy storage buffer so the control bind group has
        // something to bind when the user-supplied Control is empty.
        let external_f_dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::semi_implicit dummy external_f"),
            size: 32, // matches GpuSpatialVector
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            device,
            queue,
            kernel,
            external_f_dummy,
        }
    }
}

/// Semi-implicit Euler integrator. Construct via [`Self::new_cpu`] or
/// [`Self::new_gpu`].
pub struct SemiImplicitSolver {
    backend: SemiImplicitBackend,
}

impl Default for SemiImplicitSolver {
    fn default() -> Self {
        Self::new_cpu()
    }
}

impl SemiImplicitSolver {
    /// CPU backend.
    #[must_use]
    pub fn new_cpu() -> Self {
        Self {
            backend: SemiImplicitBackend::Cpu,
        }
    }

    /// Backwards-compatible alias for [`Self::new_cpu`].
    #[must_use]
    pub fn new() -> Self {
        Self::new_cpu()
    }

    /// GPU backend. The provided [`crate::compute::Engine`] must have
    /// been constructed with `Engine::new_gpu()`.
    #[cfg(feature = "gpu")]
    pub fn new_gpu(engine: &crate::compute::Engine) -> Result<Self> {
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("engine has no GPU device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("engine has no GPU queue".into()))?
            .clone();
        Ok(Self {
            backend: SemiImplicitBackend::Gpu(GpuBackend::new(device, queue)),
        })
    }

    /// `true` if the solver runs on a GPU backend.
    #[must_use]
    pub fn is_gpu(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            matches!(self.backend, SemiImplicitBackend::Gpu(_))
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }
}

impl Solver for SemiImplicitSolver {
    fn step(
        &mut self,
        model: &Model,
        state_in: &State,
        control: &Control,
        contacts: &Contacts,
        dt: Real,
        state_out: &mut State,
    ) -> Result<()> {
        match &mut self.backend {
            SemiImplicitBackend::Cpu => cpu_step(model, state_in, control, contacts, dt, state_out),
            #[cfg(feature = "gpu")]
            SemiImplicitBackend::Gpu(gpu) => {
                gpu.kernel.dispatch(
                    &gpu.device,
                    &gpu.queue,
                    model,
                    state_in,
                    control,
                    &gpu.external_f_dummy,
                    state_out,
                    dt,
                )
            }
        }
    }
}

fn cpu_step(
    model: &Model,
    state_in: &State,
    control: &Control,
    _contacts: &Contacts,
    dt: Real,
    state_out: &mut State,
) -> Result<()> {
    if state_in.body_q.len() != model.body_count || state_out.body_q.len() != model.body_count {
        return Err(Error::Solver(format!(
            "state size mismatch: model has {} bodies, state_in has {}, state_out has {}",
            model.body_count,
            state_in.body_q.len(),
            state_out.body_q.len()
        )));
    }

    let q_in = state_in.body_q.host()?;
    let qd_in = state_in.body_qd.host()?;
    let f_in = state_in.body_f.host()?;
    let mass = model.body_mass.host()?;
    let inv_mass = model.body_inv_mass.host()?;
    let inertia = model.body_inertia.host()?;
    let inv_inertia = model.body_inv_inertia.host()?;
    let com = model.body_com.host()?;
    let flags = model.body_flags.host()?;
    let world_idx = model.body_world.host()?;
    let gravity = model.gravity.host()?;
    let external = if control.body_external_f.len() == model.body_count {
        Some(control.body_external_f.host()?)
    } else {
        None
    };

    let mut q_out = vec![Transform::IDENTITY; model.body_count];
    let mut qd_out = vec![SpatialVector::ZERO; model.body_count];

    for tid in 0..model.body_count {
        if (flags[tid] & BodyFlags::KINEMATIC) != 0 {
            q_out[tid] = q_in[tid];
            qd_out[tid] = qd_in[tid];
            continue;
        }

        let mut applied = f_in[tid];
        if let Some(ext) = external {
            applied = applied + ext[tid];
        }

        let world = world_idx[tid].max(0) as usize;
        let g = gravity[world];

        let (q_new, qd_new) = integrate_rigid_body(
            q_in[tid],
            qd_in[tid],
            applied,
            com[tid],
            mass[tid],
            inertia[tid],
            inv_mass[tid],
            inv_inertia[tid],
            g,
            model.angular_damping,
            dt,
        );
        q_out[tid] = q_new;
        qd_out[tid] = qd_new;
    }

    state_out.body_q.host_mut()?.copy_from_slice(&q_out);
    state_out.body_qd.host_mut()?.copy_from_slice(&qd_out);
    for f in state_out.body_f.host_mut()? {
        *f = SpatialVector::ZERO;
    }
    Ok(())
}

/// Closed-form semi-implicit Euler step for one rigid body.
///
/// Direct port of Newton's `integrate_rigid_body`.
#[allow(clippy::too_many_arguments, reason = "matches Newton's reference signature")]
fn integrate_rigid_body(
    q: Transform,
    qd: SpatialVector,
    f: SpatialVector,
    com: Vec3,
    _mass: Real,
    inertia: Mat3,
    inv_mass: Real,
    inv_inertia: Mat3,
    gravity: Vec3,
    angular_damping: Real,
    dt: Real,
) -> (Transform, SpatialVector) {
    let x0 = q.position;
    let r0 = q.rotation;

    let v0 = qd.linear;
    let w0 = qd.angular;
    let f0 = f.linear;
    let t0 = f.angular;

    let x_com = x0 + r0 * com;

    let gravity_gate = if inv_mass > 0.0 { 1.0 } else { 0.0 };
    let v1 = v0 + (f0 * inv_mass + gravity * gravity_gate) * dt;
    let x1 = x_com + v1 * dt;

    let r0_inv = r0.conjugate();
    let wb = r0_inv * w0;
    let tb = r0_inv * t0 - (wb).cross(inertia * wb);

    let mut w1 = r0 * (wb + inv_inertia * tb * dt);

    let w_quat = Quat::from_xyzw(w1.x, w1.y, w1.z, 0.0);
    let r1 = (r0 + w_quat * r0 * (0.5 * dt)).normalize();

    w1 *= 1.0 - angular_damping * dt;

    let position_new = x1 - r1 * com;

    (Transform::new(position_new, r1), SpatialVector::new(v1, w1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::SpatialInertia;
    use crate::sim::{BodyParams, ModelBuilder};

    #[test]
    fn free_fall_matches_closed_form() {
        let mut b = ModelBuilder::new();
        b.set_angular_damping(0.0);
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.5))
                .with_position(Vec3::new(0.0, 10.0, 0.0)),
        );
        b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();
        let model = b.build().unwrap();

        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let contacts = Contacts::with_capacity(0);

        let mut solver = SemiImplicitSolver::new_cpu();
        let dt = 0.01;
        let n_steps = 100usize;

        for _ in 0..n_steps {
            solver
                .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        }

        let g = -crate::math::GRAVITY;
        let n = n_steps as Real;
        let expected_v = g * n * dt;
        let expected_x = 10.0 + g * dt * dt * n * (n + 1.0) / 2.0;

        let q = s_a.body_q.host().unwrap();
        let qd = s_a.body_qd.host().unwrap();
        assert!((q[0].position.y - expected_x).abs() < 1e-9);
        assert!((qd[0].linear.y - expected_v).abs() < 1e-9);
    }

    #[test]
    fn kinematic_body_passes_through() {
        let mut b = ModelBuilder::new();
        let body = b.add_body(
            BodyParams::new()
                .with_position(Vec3::new(0.0, 5.0, 0.0))
                .with_flags(BodyFlags::KINEMATIC),
        );
        b.add_shape_sphere(body, 0.1, Transform::IDENTITY).unwrap();
        let model = b.build().unwrap();

        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver = SemiImplicitSolver::new_cpu();

        for _ in 0..10 {
            solver
                .step(&model, &s_a, &control, &contacts, 0.01, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        }
        let q = s_a.body_q.host().unwrap();
        assert!((q[0].position.y - 5.0).abs() < 1e-12);
    }

    #[test]
    fn zero_mass_body_does_not_fall() {
        let mut b = ModelBuilder::new();
        let body = b.add_body(
            BodyParams::new()
                .with_mass(0.0)
                .with_inertia(Mat3::ZERO)
                .with_position(Vec3::new(0.0, 5.0, 0.0)),
        );
        b.add_shape_sphere(body, 0.1, Transform::IDENTITY).unwrap();
        let model = b.build().unwrap();

        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver = SemiImplicitSolver::new_cpu();

        for _ in 0..50 {
            solver
                .step(&model, &s_a, &control, &contacts, 0.01, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        }
        let q = s_a.body_q.host().unwrap();
        assert!((q[0].position.y - 5.0).abs() < 1e-12);
    }
}
