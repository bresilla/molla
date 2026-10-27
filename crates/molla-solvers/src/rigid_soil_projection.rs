//! Retained free-prediction coupling for one unrestricted articulated island.

use super::*;

pub(super) fn composed_submissions(#[cfg(test)] force_separate: bool) -> bool {
    let enabled = cfg!(feature = "batched-soil-projection");
    #[cfg(test)]
    let enabled = enabled && !force_separate;
    enabled
}

#[test]
fn projected_submission_selection_matches_build() {
    assert_eq!(
        composed_submissions(false),
        cfg!(feature = "batched-soil-projection")
    );
    assert!(!composed_submissions(true));
}
use crate::mpm::{
    MpmMassCouplingInputs, MpmUnilateralConfig, MpmUnilateralInputs, PreparedMpmMassCoupling,
    PreparedMpmUnilateralCoupling,
};
use crate::soil_contact_projection_gpu::SoilPairedRatePublicationInputs;
use crate::soil_contact_projection_gpu::{
    SoilBoundaryScan, SoilRatePublicationGpu, SoilRatePublicationInputs,
};
use crate::soil_rate_encoding::RateEncoding;
use molla_kernels::{
    JacobianDispatch, JacobianKernel, JacobianRequest, MassMatrixDispatch, MassMatrixKernel,
};
use molla_sim::{BodyFlags, JointType};

#[cfg(test)]
#[path = "rigid_soil_projection_budget_tests.rs"]
mod budget_tests;

fn com_copy_ranges(bodies: &[u32], bytes_per_body: u64) -> Vec<[u64; 3]> {
    let mut ranges: Vec<[u64; 3]> = Vec::new();
    for (slot, &body) in bodies.iter().enumerate() {
        let source = u64::from(body) * bytes_per_body;
        let destination = slot as u64 * bytes_per_body;
        if let Some(last) = ranges.last_mut()
            && last[0] + last[2] == source
            && last[1] + last[2] == destination
        {
            last[2] += bytes_per_body;
        } else {
            ranges.push([source, destination, bytes_per_body]);
        }
    }
    ranges
}

#[test]
fn native_com_copy_ranges_preserve_binding_order() {
    assert_eq!(com_copy_ranges(&[1, 2, 3, 4, 5], 240), [[240, 0, 1200]]);
    assert_eq!(
        com_copy_ranges(&[5, 1, 2, 0, 5], 240),
        [
            [1200, 0, 240],
            [240, 240, 480],
            [0, 720, 240],
            [1200, 960, 240]
        ]
    );
    assert!(com_copy_ranges(&[], 240).is_empty());
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Mode {
    Sticking {
        nodes_per_body: u32,
        boundary_scan: SoilBoundaryScan,
    },
    Unilateral(MpmUnilateralConfig),
    Admm {
        config: MpmUnilateralConfig,
        coefficients: [f32; 64],
    },
    Friction {
        config: MpmUnilateralConfig,
        coefficients: [f32; 64],
        fixed_walls: bool,
        newton: bool,
    },
}

enum Coupling {
    Sticking(PreparedMpmMassCoupling),
    Unilateral(PreparedMpmUnilateralCoupling),
}

impl Mode {
    fn coupling_bytes(
        self,
        soil: &MpmSimulation,
        dofs: u32,
        bodies: u32,
        encoding: RateEncoding,
    ) -> Result<u64> {
        if let Self::Sticking { nodes_per_body, .. } = self {
            return PreparedMpmMassCoupling::required_bytes(soil, dofs, bodies, nodes_per_body);
        }
        PreparedMpmUnilateralCoupling::required_bytes_for_grid(soil, |size| match self {
            Self::Unilateral(config) => {
                PreparedMpmUnilateralCoupling::required_bytes(config, dofs, size, bodies)
            }
            Self::Admm { config, .. } => {
                if encoding != RateEncoding::Paired {
                    return Err(invalid("ADMM soil requires paired machine rates"));
                }
                PreparedMpmUnilateralCoupling::required_admm_bytes(config, dofs, size, bodies)
            }
            Self::Friction {
                config,
                newton: true,
                ..
            } => PreparedMpmUnilateralCoupling::required_newton_bytes(config, dofs, size, bodies),
            Self::Friction {
                config,
                fixed_walls: true,
                ..
            } => PreparedMpmUnilateralCoupling::required_wall_friction_bytes_for_rates(
                config, dofs, size, bodies, encoding,
            ),
            Self::Friction {
                config,
                fixed_walls: false,
                ..
            } => PreparedMpmUnilateralCoupling::required_friction_bytes(config, dofs, size, bodies),
            Self::Sticking { .. } => unreachable!(),
        })
    }
}
impl Coupling {
    fn newton_seed_buffer(&self) -> Option<&wgpu::Buffer> {
        match self {
            Self::Unilateral(p) => p.newton_seed_buffer(),
            Self::Sticking(_) => None,
        }
    }
    fn newton_failure_buffer(&self) -> Option<&wgpu::Buffer> {
        match self {
            Self::Sticking(_) => None,
            Self::Unilateral(p) => p.newton_failure_buffer(),
        }
    }
    fn newton_preparation_failure_buffer(&self) -> Option<&wgpu::Buffer> {
        match self {
            Self::Sticking(_) => None,
            Self::Unilateral(p) => p.newton_preparation_failure_buffer(),
        }
    }
    fn admm_preparation_failure_buffer(&self) -> Option<&wgpu::Buffer> {
        match self {
            Self::Sticking(_) => None,
            Self::Unilateral(p) => p.admm_preparation_failure_buffer(),
        }
    }
    fn failure_solver(
        &self,
    ) -> Option<&crate::soil_shared_wall_friction_gpu::ExperimentalSoilSharedWallFrictionGpu> {
        match self {
            Self::Sticking(_) => None,
            Self::Unilateral(p) => p.failure_solver(),
        }
    }
    fn local_inputs_buffer(&self) -> Option<&wgpu::Buffer> {
        match self {
            Self::Sticking(_) => None,
            Self::Unilateral(p) => p.local_inputs_buffer(),
        }
    }
    fn candidate_failure_buffer(&self) -> Option<&wgpu::Buffer> {
        match self {
            Self::Sticking(_) => None,
            Self::Unilateral(p) => Some(p.candidate_failure_buffer()),
        }
    }
    fn status_buffer(&self) -> &wgpu::Buffer {
        match self {
            Self::Sticking(p) => p.status_buffer(),
            Self::Unilateral(p) => p.status_buffer(),
        }
    }
    fn step(&mut self, soil: &mut MpmSimulation, dt: f32) -> Result<()> {
        match self {
            Self::Sticking(p) => p.step(soil, dt),
            Self::Unilateral(p) => p.step(soil, dt),
        }
    }
    fn step_with_commands(
        &mut self,
        soil: &mut MpmSimulation,
        dt: f32,
        before: impl FnOnce(&mut wgpu::CommandEncoder),
        after: impl FnOnce(&mut wgpu::CommandEncoder),
    ) -> Result<()> {
        match self {
            Self::Sticking(p) => p.step_with_commands(soil, dt, before, after),
            Self::Unilateral(p) => p.step_with_commands(soil, dt, before, after),
        }
    }
    fn prepare_machine_publication(
        &self,
        input: SoilRatePublicationInputs<'_>,
    ) -> Result<SoilRatePublicationGpu> {
        match self {
            Self::Sticking(p) => p.prepare_machine_publication(input),
            Self::Unilateral(p) => p.prepare_machine_publication(input),
        }
    }
    fn prepare_paired_machine_publication(
        &self,
        input: SoilPairedRatePublicationInputs<'_>,
    ) -> Result<SoilRatePublicationGpu> {
        match self {
            Self::Unilateral(p) => p.prepare_paired_machine_publication(input),
            Self::Sticking(_) => Err(invalid("sticking coupling does not support paired rates")),
        }
    }
    #[cfg(test)]
    fn trace_geometry(&self) -> [wgpu::Buffer; 3] {
        match self {
            Self::Sticking(p) => p.trace_geometry(),
            Self::Unilateral(_) => panic!("sticking trace unavailable for unilateral coupling"),
        }
    }
    #[cfg(test)]
    fn prepare_projection_replay_capture(
        &mut self,
    ) -> crate::soil_contact_projection_gpu::SoilProjectionReplay {
        match self {
            Self::Sticking(p) => p.prepare_projection_replay_capture(),
            Self::Unilateral(_) => panic!("sticking replay unavailable for unilateral coupling"),
        }
    }
}

pub(super) struct ProjectedSoil {
    mass: Option<[MassMatrixDispatch; 2]>,
    shared_mass: Option<[wgpu::Buffer; 2]>,
    jacobians: Option<[JacobianDispatch; 2]>,
    shared_jacobians: Option<[wgpu::Buffer; 2]>,
    jacobian_buffer: wgpu::Buffer,
    jacobian_copies: Vec<[u64; 3]>,
    coupling: [Coupling; 2],
    mode: Mode,
    publication: [SoilRatePublicationGpu; 2],
    guard_pipeline: wgpu::ComputePipeline,
    guard_groups: [wgpu::BindGroup; 2],
    status_readback: wgpu::Buffer,
    bytes: u64,
    last_input: Option<usize>,
    #[cfg(test)]
    pub(super) newton_work_trace: Option<super::projected_tests::newton_work::Trace>,
    #[cfg(test)]
    pub(super) admm_work_trace: Option<super::projected_tests::admm_work::Trace>,
    #[cfg(test)]
    pub(super) submission_counts: [u64; 2],
}

impl ProjectedSoil {
    pub(super) fn newton_method_name(&self) -> Option<&'static str> {
        let names = self.coupling.each_ref().map(|owner| match owner {
            Coupling::Unilateral(p) => p.newton_method_name(),
            Coupling::Sticking(_) => None,
        });
        if names[0] == names[1] { names[0] } else { None }
    }
    pub(super) fn indexed_sort_backend(&self) -> Option<&'static str> {
        match &self.coupling[0] {
            Coupling::Unilateral(owner) => owner.indexed_sort_backend(),
            Coupling::Sticking(_) => None,
        }
    }
    #[cfg(test)]
    pub(super) fn native_jacobian_copy_work(&self) -> (usize, u64) {
        (
            self.jacobian_copies.len(),
            self.jacobian_copies.iter().map(|r| r[2]).sum(),
        )
    }
    #[cfg(test)]
    pub(super) fn shares_native_operators(&self) -> bool {
        self.mass.is_none()
            && self.shared_mass.is_some()
            && self.jacobians.is_none()
            && self.shared_jacobians.is_some()
    }
    pub(super) fn matches_operators(
        &self,
        mass: Option<[&wgpu::Buffer; 2]>,
        jacobians: Option<[&wgpu::Buffer; 2]>,
    ) -> bool {
        self.shared_mass.as_ref().map(|m| m.each_ref()) == mass
            && self.shared_jacobians.as_ref().map(|j| j.each_ref()) == jacobians
    }
    pub(super) fn invalidate_newton_history(&self, queue: &wgpu::Queue) {
        for coupling in &self.coupling {
            if let Coupling::Unilateral(owner) = coupling {
                owner.invalidate_newton_history(queue);
            }
        }
    }
    #[cfg(test)]
    pub(super) fn trace_wall_stages(&mut self, path: &std::path::Path) {
        for (parity, coupling) in self.coupling.iter_mut().enumerate() {
            match coupling {
                Coupling::Unilateral(p) => {
                    p.trace_wall_stages(path.join(format!("parity-{parity}")))
                }
                _ => panic!("wall trace requires shared wall friction"),
            }
        }
    }
    #[cfg(test)]
    pub(super) fn capture_first_wall_input(
        &mut self,
    ) -> [crate::soil_shared_wall_friction_gpu::capture::SharedWallSnapshot; 2] {
        self.coupling.each_mut().map(|coupling| match coupling {
            Coupling::Unilateral(p) => p.capture_first_wall_input(),
            _ => panic!("first wall input capture requires shared wall friction"),
        })
    }
    #[cfg(test)]
    pub(super) fn wall_trace(&self) -> [[wgpu::Buffer; 5]; 2] {
        self.coupling.each_ref().map(|c| match c {
            Coupling::Unilateral(p) => p.wall_trace(),
            _ => panic!("wall trace requires shared wall friction"),
        })
    }
    #[cfg(test)]
    pub(super) fn prepare_normal_reference_check(&mut self, source: &str) -> [wgpu::Buffer; 2] {
        self.coupling.each_mut().map(|coupling| match coupling {
            Coupling::Unilateral(p) => p.prepare_reference_check(source),
            Coupling::Sticking(_) => panic!("normal comparison requires unilateral mode"),
        })
    }

    #[cfg(test)]
    pub(super) fn prepare_normal_failure_capture(
        &mut self,
    ) -> [crate::soil_normal_projection_gpu::replay::NormalReplay; 2] {
        self.coupling.each_mut().map(|coupling| match coupling {
            Coupling::Unilateral(p) => p.prepare_failure_capture(),
            Coupling::Sticking(_) => panic!("normal capture requires unilateral mode"),
        })
    }
    #[cfg(test)]
    pub(super) fn trace_geometry(&self) -> [[wgpu::Buffer; 3]; 2] {
        self.coupling
            .each_ref()
            .map(|coupling| coupling.trace_geometry())
    }

    #[cfg(test)]
    pub(super) fn boundary_scan(&self) -> SoilBoundaryScan {
        match self.mode {
            Mode::Sticking { boundary_scan, .. } => boundary_scan,
            Mode::Unilateral(_) | Mode::Friction { .. } | Mode::Admm { .. } => {
                SoilBoundaryScan::Full
            }
        }
    }
    pub(super) fn mode(&self) -> Mode {
        self.mode
    }
    #[cfg(test)]
    pub(super) fn prepare_replay_capture(
        &mut self,
    ) -> [crate::soil_contact_projection_gpu::SoilProjectionReplay; 2] {
        self.coupling
            .each_mut()
            .map(|coupling| coupling.prepare_projection_replay_capture())
    }

    pub(super) fn validate(scene: &RigidScene) -> Result<()> {
        let model = scene.model();
        if model.joint_count == 0
            || model.joint_count > 64
            || model.body_count != model.joint_count
            || model.articulation_count > 1
            || !(1..=32).contains(&model.joint_dof_total)
            || scene.colliders().next().is_some()
            || scene.loop_joints().next().is_some()
            || scene.breakable_joints().next().is_some()
            || model
                .body_flags
                .host()?
                .iter()
                .any(|f| f & BodyFlags::IMMOVABLE != 0)
            || model.joint_enabled.host()?.contains(&0)
            || model
                .joint_type
                .host()?
                .iter()
                .any(|&k| k == JointType::Distance as u32 || k > JointType::D6 as u32)
            || model
                .joint_limit_lower
                .host()?
                .iter()
                .any(|v| v.is_finite())
            || model
                .joint_limit_upper
                .host()?
                .iter()
                .any(|v| v.is_finite())
            || model
                .joint_velocity_limit
                .host()?
                .iter()
                .any(|v| v.is_finite())
        {
            return Err(invalid(
                "mass coupling requires one dynamic unrestricted articulation, 1..32 DOFs, no rigid colliders or loop/breakable/distance constraints",
            ));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        engine: &Arc<Engine>,
        soil: &mut MpmSimulation,
        scene: &RigidScene,
        states: &[State; 2],
        solver: &FeatherstoneSolver,
        shared_mass: Option<[&wgpu::Buffer; 2]>,
        shared_jacobians: Option<[&wgpu::Buffer; 2]>,
        bindings: &[MpmRigidDeviceBinding],
        mode: Mode,
        budget: u64,
    ) -> Result<Self> {
        Self::validate(scene)?;
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let n = scene.model().joint_dof_total as u64;
        if shared_mass.is_some() != shared_jacobians.is_some() {
            return Err(invalid(
                "native mass and Jacobians must share prediction ownership",
            ));
        }
        if shared_jacobians.is_some_and(|buffers| {
            buffers.iter().any(|buffer| {
                buffer.size() < 24 * scene.model().body_count as u64 * n
                    || !buffer.usage().contains(wgpu::BufferUsages::COPY_SRC)
            })
        }) {
            return Err(invalid("invalid native COM Jacobian buffer"));
        }
        let encoding = if solver.gpu_uses_paired_rates() {
            RateEncoding::Paired
        } else {
            RateEncoding::Scalar
        };
        if encoding == RateEncoding::Paired
            && !matches!(
                mode,
                Mode::Friction {
                    fixed_walls: true,
                    ..
                } | Mode::Admm { .. }
            )
        {
            return Err(invalid("paired machine rates require shared wall friction"));
        }
        if matches!(mode, Mode::Friction { newton: true, .. }) && encoding != RateEncoding::Paired {
            return Err(invalid("Newton soil requires paired machine rates"));
        }
        if matches!(mode, Mode::Admm { .. }) && encoding != RateEncoding::Paired {
            return Err(invalid("ADMM soil requires paired machine rates"));
        }
        let b = bindings.len() as u64;
        let coupling_bytes = mode.coupling_bytes(soil, n as u32, b as u32, encoding)?;
        let bytes = 2 * coupling_bytes
            + if shared_mass.is_some() { 0 } else { 4 * n * n }
            + 24 * b * n
            + if shared_jacobians.is_some() {
                0
            } else {
                64 * b
            }
            + if shared_mass.is_some() { 224 } else { 312 };
        if bytes > budget {
            return Err(invalid("mass coupling exceeds retained-buffer budget"));
        }
        if let Mode::Admm { config, .. } = mode {
            crate::soil_newton_prepare_gpu::ExperimentalSoilNewtonPrepareGpu::check_selected_execution(
                engine,
                config.node_capacity,
            )?;
            let limits = device.limits();
            let c = u64::from(config.node_capacity);
            let largest = (16 * c * n).max(336 * c).max(4 * n * n + 16 * n + 128);
            if limits.max_storage_buffers_per_shader_stage
                < if soil.active_grid_enabled() { 14 } else { 12 }
                || limits.max_bindings_per_bind_group < 16
                || limits.max_compute_workgroup_storage_size < 38400
                || largest > limits.max_storage_buffer_binding_size
                || largest > limits.max_buffer_size
            {
                return Err(invalid("ADMM coupling exceeds device limits"));
            }
        }
        if let Mode::Friction {
            newton: true,
            config,
            ..
        } = mode
        {
            let limits = device.limits();
            let largest = (64 * 65 * u64::from(config.node_capacity))
                .max(33280 + 432 * u64::from(config.node_capacity.max(1)))
                .max(
                    crate::soil_newton_failure::NewtonFailureCapture::required_bytes(
                        n as u32,
                        config.node_capacity,
                    )? - 16,
                );
            if limits.max_storage_buffers_per_shader_stage < 21
                || limits.max_bindings_per_bind_group < 22
                || limits.max_compute_workgroup_storage_size < 24000
                || largest > limits.max_storage_buffer_binding_size
                || largest > limits.max_buffer_size
            {
                return Err(invalid("Newton coupling exceeds device limits"));
            }
        }
        let buffer = |label, size| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }
            )
        };
        let owned_mass = shared_mass.is_none().then(|| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("soil runtime generalized mass"),
                    size: 4 * n * n,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }
            )
        });
        let jacobian_buffer = buffer("soil runtime COM Jacobians", 24 * b * n);
        let crba = if shared_mass.is_none() {
            Some(MassMatrixKernel::new(engine, scene.model())?)
        } else {
            None
        };
        let jac = if shared_jacobians.is_none() {
            Some(JacobianKernel::new(engine, scene.model())?)
        } else {
            None
        };
        let requests = bindings
            .iter()
            .enumerate()
            .map(|(slot, binding)| {
                let index = scene
                    .body_index(binding.body)
                    .ok_or_else(|| invalid("missing projected soil body"))?
                    .index();
                let com = scene.model().body_com.host()?[index];
                Ok(JacobianRequest {
                    articulation: 0,
                    target_body: index as u32,
                    output_offset: (slot as u64 * 6 * n) as u32,
                    body_q_offset: 0,
                    point_local: [com.x as f32, com.y as f32, com.z as f32, 0.0],
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let jacobian_copies = if shared_jacobians.is_some() {
            com_copy_ranges(
                &requests.iter().map(|r| r.target_body).collect::<Vec<_>>(),
                24 * n,
            )
        } else {
            Vec::new()
        };
        let mut mass = Vec::new();
        let mut jacobians = Vec::new();
        let mut coupling = Vec::new();
        let mut publication = Vec::new();
        for current in 0..2 {
            let mass_buffer = shared_mass
                .map(|m| m[current])
                .unwrap_or_else(|| owned_mass.as_ref().unwrap());
            let free_rates = match encoding {
                RateEncoding::Scalar => states[1 - current].joint_qd.device_buffer_current()?,
                RateEncoding::Paired => solver.gpu_paired_rates_buffer(&states[1 - current])?,
            };
            if let Some(crba) = &crba {
                mass.push(crba.prepare_on_device(
                    device,
                    queue,
                    scene.model(),
                    &states[current],
                    mass_buffer,
                )?);
            }
            if let Some(jac) = &jac {
                jacobians.push(
                    jac.prepare_with_body_q_on_device(
                        device,
                        queue,
                        scene.model(),
                        states[current].body_q.device_buffer_current()?,
                        &requests,
                        &jacobian_buffer,
                    )?
                    .ok_or_else(|| invalid("empty COM Jacobian requests"))?,
                );
            }
            let projected = match mode {
                Mode::Admm {
                    config,
                    coefficients,
                } => Coupling::Unilateral(PreparedMpmUnilateralCoupling::new_experimental_admm(
                    engine.clone(),
                    soil,
                    MpmUnilateralInputs {
                        mass: &mass_buffer,
                        jacobians: &jacobian_buffer,
                        free_rates,
                        dofs: n as u32,
                        max_owned_bytes: coupling_bytes,
                    },
                    config,
                    &coefficients[..bindings.len()],
                )?),
                Mode::Sticking {
                    nodes_per_body,
                    boundary_scan,
                } => Coupling::Sticking(PreparedMpmMassCoupling::new_with_boundary_scan(
                    engine.clone(),
                    soil,
                    MpmMassCouplingInputs {
                        mass: &mass_buffer,
                        jacobians: &jacobian_buffer,
                        free_rates: states[1 - current].joint_qd.device_buffer_current()?,
                        dofs: n as u32,
                        nodes_per_body,
                        max_owned_bytes: coupling_bytes,
                    },
                    boundary_scan,
                )?),
                Mode::Unilateral(config) => {
                    Coupling::Unilateral(PreparedMpmUnilateralCoupling::new(
                        engine.clone(),
                        soil,
                        MpmUnilateralInputs {
                            mass: &mass_buffer,
                            jacobians: &jacobian_buffer,
                            free_rates: states[1 - current].joint_qd.device_buffer_current()?,
                            dofs: n as u32,
                            max_owned_bytes: coupling_bytes,
                        },
                        config,
                    )?)
                }
                Mode::Friction {
                    config,
                    coefficients,
                    fixed_walls,
                    newton,
                } => Coupling::Unilateral(if newton {
                    PreparedMpmUnilateralCoupling::new_with_newton_rates(
                        engine.clone(),
                        soil,
                        MpmUnilateralInputs {
                            mass: &mass_buffer,
                            jacobians: &jacobian_buffer,
                            free_rates,
                            dofs: n as u32,
                            max_owned_bytes: coupling_bytes,
                        },
                        config,
                        &coefficients[..bindings.len()],
                    )?
                } else {
                    PreparedMpmUnilateralCoupling::new_with_friction_rates(
                        engine.clone(),
                        soil,
                        MpmUnilateralInputs {
                            mass: &mass_buffer,
                            jacobians: &jacobian_buffer,
                            free_rates,
                            dofs: n as u32,
                            max_owned_bytes: coupling_bytes,
                        },
                        config,
                        &coefficients[..bindings.len()],
                        fixed_walls,
                        encoding,
                    )?
                }),
            };
            publication.push(if encoding == RateEncoding::Paired {
                projected.prepare_paired_machine_publication(SoilPairedRatePublicationInputs {
                    rates: free_rates,
                    velocity_delta: solver
                        .gpu_paired_velocity_delta_buffer()
                        .ok_or_else(|| invalid("missing paired machine delta"))?,
                    machine_status: solver
                        .gpu_status_buffer()
                        .ok_or_else(|| invalid("missing machine status"))?,
                    max_owned_bytes: 16,
                })?
            } else {
                projected.prepare_machine_publication(SoilRatePublicationInputs {
                    rates: states[1 - current].joint_qd.device_buffer_current()?,
                    velocity_delta: solver
                        .gpu_velocity_delta_buffer()
                        .ok_or_else(|| invalid("missing machine delta"))?,
                    machine_status: solver
                        .gpu_status_buffer()
                        .ok_or_else(|| invalid("missing machine status"))?,
                    max_owned_bytes: 16,
                })?
            });
            coupling.push(projected);
        }
        if cfg!(feature = "adjacent-newton") {
            let (first, second) = coupling.split_at_mut(1);
            if let (Coupling::Unilateral(a), Coupling::Unilateral(b)) =
                (&mut first[0], &mut second[0])
            {
                a.use_previous_newton_history(b)?;
                b.use_previous_newton_history(a)?;
            }
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("soil upstream machine guard"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/soil_machine_guard.wgsl").into(),
            ),
        });
        let guard_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("soil upstream machine guard"),
            layout: None,
            module: &shader,
            entry_point: Some("guard"),
            compilation_options: Default::default(),
            cache: None,
        });
        let guard_groups = std::array::from_fn(|i| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("soil upstream machine guard"),
                layout: &guard_pipeline.get_bind_group_layout(0),
                entries: &[
                    solver.gpu_status_buffer().unwrap(),
                    coupling[i].status_buffer(),
                ]
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
            })
        });
        let status_readback = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("soil projection failure status"),
                size: 192,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        Ok(Self {
            mode,
            mass: if shared_mass.is_none() {
                Some(mass.try_into().ok().unwrap())
            } else {
                None
            },
            shared_mass: shared_mass.map(|m| m.map(Clone::clone)),
            jacobians: if shared_jacobians.is_none() {
                Some(jacobians.try_into().ok().unwrap())
            } else {
                None
            },
            shared_jacobians: shared_jacobians.map(|j| j.map(Clone::clone)),
            jacobian_buffer,
            jacobian_copies,
            coupling: coupling.try_into().ok().unwrap(),
            publication: publication.try_into().ok().unwrap(),
            guard_pipeline,
            guard_groups,
            status_readback,
            bytes,
            last_input: None,
            #[cfg(test)]
            newton_work_trace: None,
            #[cfg(test)]
            admm_work_trace: None,
            #[cfg(test)]
            submission_counts: [0; 2],
        })
    }

    pub(super) fn bytes(&self) -> u64 {
        self.bytes
    }

    #[cfg(test)]
    pub(super) fn enable_admm_work_trace(
        &mut self,
        device: &wgpu::Device,
        capacity: usize,
    ) -> Result<()> {
        if self.last_input.is_some() || self.admm_work_trace.is_some() {
            return Err(invalid("ADMM trace must precede stepping"));
        }
        let source = |i| match &self.coupling[i] {
            Coupling::Unilateral(p) => p
                .admm_trace_sources()
                .ok_or_else(|| invalid("ADMM trace requires ADMM")),
            _ => Err(invalid("ADMM trace requires ADMM")),
        };
        let (n, a) = source(0)?;
        let (m, b) = source(1)?;
        if n != m {
            return Err(invalid("ADMM trace widths differ"));
        }
        self.admm_work_trace = Some(super::projected_tests::admm_work::Trace::new(
            device,
            n,
            capacity,
            [a, b],
        )?);
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn admm_status_buffers(&self) -> [wgpu::Buffer; 2] {
        assert!(matches!(self.mode, Mode::Admm { .. }));
        self.coupling.each_ref().map(|p| p.status_buffer().clone())
    }

    #[cfg(test)]
    pub(super) fn newton_success_buffers(&self) -> [wgpu::Buffer; 2] {
        std::array::from_fn(|i| self.coupling[i].newton_failure_buffer().unwrap().clone())
    }
    #[cfg(test)]
    pub(super) fn newton_success_seeds(&self) -> [Option<wgpu::Buffer>; 2] {
        self.coupling.each_ref().map(|c| match c {
            Coupling::Unilateral(p) => p.newton_success_seed_buffer().cloned(),
            _ => None,
        })
    }
    #[cfg(all(
        test,
        feature = "active-publication-newton",
        not(target_arch = "wasm32")
    ))]
    pub(super) fn newton_shadow_buffers(&self) -> [wgpu::Buffer; 2] {
        self.coupling.each_ref().map(|c| match c {
            Coupling::Unilateral(p) => p.newton_shadow_buffer().unwrap().clone(),
            _ => panic!("Newton shadow requires unilateral coupling"),
        })
    }

    pub(super) fn fault_detail(
        &self,
        engine: &Engine,
        completion: &soil_completion::CompletionDeadline<'_>,
    ) -> Result<String> {
        let device = engine.device().unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&self.status_readback, 0, None);
        for i in 0..2 {
            encoder.copy_buffer_to_buffer(
                self.coupling[i].status_buffer(),
                0,
                &self.status_readback,
                i as u64 * 96,
                if self.coupling[i].local_inputs_buffer().is_some()
                    || matches!(
                        self.mode,
                        Mode::Friction { newton: true, .. } | Mode::Admm { .. }
                    )
                {
                    32
                } else {
                    4
                },
            );
            if let Some(buffer) = self.coupling[i].candidate_failure_buffer() {
                encoder.copy_buffer_to_buffer(
                    buffer,
                    4,
                    &self.status_readback,
                    i as u64 * 96 + 32,
                    64,
                );
            }
        }
        engine.queue().unwrap().submit([encoder.finish()]);
        map_read_with_completion(device, &self.status_readback, Some(completion))?;
        let mapped = molla_core::gpu_readback!(self.status_readback.slice(..));
        let words = bytemuck::pod_read_unaligned::<[u32; 48]>(&mapped);
        let values = [words[0], words[24]];
        drop(mapped);
        self.status_readback.unmap();
        let names = |value| {
            let labels: &[(u32, &str)] = if matches!(self.mode, Mode::Admm { .. }) {
                &[
                    (1, "input-or-upstream"),
                    (2, "mass-not-SPD"),
                    (4, "nonfinite"),
                    (128, "geometry-or-friction-input"),
                    (256, "candidate-input"),
                    (512, "candidate-capacity"),
                    (1024, "multiple-body-node"),
                    (2048, "candidate-coverage"),
                    (4096, "admm-preparation"),
                    (8192, "admm-solve"),
                    (16384, "admm-publication"),
                ]
            } else if matches!(
                self.mode,
                Mode::Friction {
                    fixed_walls: true,
                    ..
                }
            ) {
                &[
                    (1, "input-or-upstream"),
                    (2, "mass-not-SPD"),
                    (4, "nonfinite"),
                    (16, "shared-wall-sweep-limit"),
                    (4096, "newton-preparation"),
                    (8192, "newton-solve"),
                    (16384, "newton-publication"),
                    (64, "shared-wall-local-solve"),
                    (128, "geometry-or-friction-input"),
                    (256, "candidate-input"),
                    (512, "candidate-capacity"),
                    (1024, "multiple-body-node"),
                    (2048, "candidate-coverage"),
                ]
            } else if matches!(self.mode, Mode::Friction { .. }) {
                &[
                    (1, "input-or-upstream"),
                    (2, "mass-not-SPD"),
                    (4, "nonfinite"),
                    (8, "line-search"),
                    (16, "iterations"),
                    (32, "friction-jacobian"),
                    (64, "fixed-wall-friction-unsupported"),
                    (128, "geometry-or-friction-input"),
                    (256, "candidate-input"),
                    (512, "candidate-capacity"),
                    (1024, "multiple-body-node"),
                    (2048, "candidate-coverage"),
                ]
            } else if matches!(self.mode, Mode::Unilateral(_)) {
                &[
                    (1, "input-or-upstream"),
                    (2, "mass-not-SPD"),
                    (4, "nonfinite"),
                    (8, "line-search"),
                    (16, "iterations"),
                    (32, "floor-input"),
                    (64, "hard-system"),
                    (128, "geometry-input"),
                    (256, "candidate-input"),
                    (512, "candidate-capacity"),
                    (1024, "multiple-body-node"),
                    (2048, "candidate-coverage"),
                ]
            } else {
                &[
                    (1, "input-or-upstream"),
                    (2, "mass-not-SPD"),
                    (4, "nonfinite"),
                    (8, "residual"),
                    (16, "node-capacity"),
                    (32, "fixed-grid-boundary"),
                    (64, "boundary-solve"),
                    (128, "boundary-basis"),
                    (256, "boundary-iterations"),
                    (512, "boundary-dual"),
                ]
            };
            labels
                .iter()
                .copied()
                .chain(std::iter::once((
                    crate::mpm::MPM_COUPLING_PARTICLE_FAULT,
                    "particle-grid-fault",
                )))
                .filter_map(|(bit, name)| (value & bit != 0).then_some(name))
                .collect::<Vec<_>>()
                .join("|")
        };
        let mut detail = format!(
            "soil projection input-state 0={}({}), 1={}({})",
            values[0],
            names(values[0]),
            values[1],
            names(values[1])
        );
        for i in 0..2 {
            let witness = &words[i * 24 + 8..i * 24 + 24];
            if values[i] != 0 && matches!(self.mode, Mode::Friction { newton: true, .. }) {
                detail.push_str(&format!("; newton-{i} [fault, preparation, solve, publication, steps, inner, trials, pad]={:?}",&words[i*24..i*24+8]));
            }
            if values[i] != 0 && matches!(self.mode, Mode::Admm { .. }) {
                detail.push_str(&format!("; admm-{i} [fault, preparation, solve, publication, steps, factors, reserved, reserved]={:?}", &words[i*24..i*24+8]));
            }
            if values[i] & 4096 != 0
                && words[i * 24 + 1] != 0
                && let Some(buffer) = self.coupling[i]
                    .newton_preparation_failure_buffer()
                    .or_else(|| self.coupling[i].admm_preparation_failure_buffer())
                && let Some(root) =
                    std::env::var_os("MOLLA_SOIL_FAILURE_CAPTURE").filter(|p| !p.is_empty())
            {
                let root = std::path::PathBuf::from(root);
                let method = if matches!(self.mode, Mode::Admm { .. }) {
                    "admm"
                } else {
                    "newton"
                };
                let path = root.join(format!("{method}-preparation-{i}"));
                let result = std::fs::create_dir_all(&root)
                    .map_err(|e| invalid(&e.to_string()))
                    .and_then(|_| {
                        crate::soil_newton_prepare_gpu::capture_failure(
                            engine,
                            buffer,
                            &path,
                            |staging| map_read_with_completion(device, staging, Some(completion)),
                        )
                    });
                match result {
                    Ok(()) => {
                        detail.push_str(&format!("; preparation_snapshot={}", path.display()))
                    }
                    Err(error) => detail.push_str(&format!("; preparation_snapshot_error={error}")),
                }
            }
            if values[i] & 8192 != 0
                && words[i * 24 + 1] == 0
                && words[i * 24 + 2] != 0
                && let Some(buffer) = self.coupling[i].newton_failure_buffer()
                && let Some(root) =
                    std::env::var_os("MOLLA_SOIL_FAILURE_CAPTURE").filter(|p| !p.is_empty())
            {
                let root = std::path::PathBuf::from(root);
                let path = root.join(format!("newton-solve-{i}"));
                let result = std::fs::create_dir_all(&root)
                    .map_err(|e| invalid(&e.to_string()))
                    .and_then(|_| {
                        crate::soil_newton_failure::capture_failure_with_seed(
                            engine,
                            buffer,
                            self.coupling[i].newton_seed_buffer(),
                            &path,
                            |staging| map_read_with_completion(device, staging, Some(completion)),
                        )
                    });
                match result {
                    Ok(()) => detail.push_str(&format!("; newton_snapshot={}", path.display())),
                    Err(error) => detail.push_str(&format!("; newton_snapshot_error={error}")),
                }
            }
            if values[i] != 0
                && let Some(solver) = self.coupling[i].failure_solver()
                && let Some(root) =
                    std::env::var_os("MOLLA_SOIL_FAILURE_CAPTURE").filter(|p| !p.is_empty())
            {
                let root = std::path::PathBuf::from(root);
                let path = root.join(format!("parity-{i}"));
                let capture = std::fs::create_dir_all(&root)
                    .map_err(|e| invalid(&e.to_string()))
                    .and_then(|_| {
                        solver.capture_failure(engine, &path, |buffer| {
                            map_read_with_completion(device, buffer, Some(completion))
                        })
                    });
                match capture {
                    Ok(()) => detail.push_str(&format!("; shared_snapshot={}", path.display())),
                    Err(error) => detail.push_str(&format!("; shared_snapshot_error={error}")),
                }
            }
            if values[i] & 2048 != 0 && witness[0] != 0 {
                let values = witness[8..12]
                    .iter()
                    .copied()
                    .map(f32::from_bits)
                    .collect::<Vec<_>>();
                let point_mass = witness[12..16]
                    .iter()
                    .copied()
                    .map(f32::from_bits)
                    .collect::<Vec<_>>();
                detail.push_str(&format!("; coverage[{i}] claims={} reason={} node={} body={} owner={} slot={} walls={} dt={:?} values={values:?} point_mass={point_mass:?}",
                    witness[0], witness[1], witness[2], witness[3], witness[4], witness[5], witness[6], f32::from_bits(witness[7])));
            }
            if let Some(buffer) = self.coupling[i].local_inputs_buffer() {
                if values[i] & (4 | 16 | 64) == 0 {
                    continue;
                }
                let status = &words[i * 24..i * 24 + 8];
                detail.push_str(&format!("; local[{i}] status={status:?}"));
                if status[7] != 0 {
                    let reason = match status[7] & 255 {
                        1 => "machine-momentum",
                        2 => "normal-or-friction-cone",
                        3 => "friction-slip",
                        4 => "inactive-wall",
                        5 => "wall-complementarity",
                        6 => "energy",
                        7 => "soil-momentum",
                        8 => "normal-impulse",
                        9 => "normal-feasibility",
                        10 => "normal-complementarity",
                        11 => "friction-cone",
                        _ => "unknown",
                    };
                    detail.push_str(&format!(
                        " certificate={reason} index_plus_one={}",
                        status[7] >> 8
                    ));
                }
                let contact = status[3];
                if contact == 0 || u64::from(contact) * 112 > buffer.size() {
                    continue;
                }
                let mut encoder = device.create_command_encoder(&Default::default());
                encoder.copy_buffer_to_buffer(
                    buffer,
                    u64::from(contact - 1) * 112,
                    &self.status_readback,
                    0,
                    112,
                );
                engine.queue().unwrap().submit([encoder.finish()]);
                map_read_with_completion(device, &self.status_readback, Some(completion))?;
                let mapped = molla_core::gpu_readback!(self.status_readback.slice(..));
                let input = bytemuck::pod_read_unaligned::<[u32; 28]>(&mapped[..112]);
                drop(mapped);
                self.status_readback.unmap();
                detail.push_str(&format!(" local_input_words={input:?}"));
            }
        }
        Ok(detail)
    }

    pub(super) fn prepare_input(&self, encoder: &mut wgpu::CommandEncoder, current: usize) {
        if let Some(mass) = &self.mass {
            mass[current].encode(encoder);
        }
        if let Some(jacobians) = &self.jacobians {
            jacobians[current].encode(encoder);
        }
    }

    pub(super) fn newton_work_status(&self, current: usize) -> Option<&wgpu::Buffer> {
        matches!(self.mode, Mode::Friction { newton: true, .. })
            .then(|| self.coupling[current].status_buffer())
    }

    pub(super) fn step(
        &mut self,
        _engine: &Engine,
        soil: &mut MpmSimulation,
        current: usize,
        dt: f32,
        after: impl FnOnce(&mut wgpu::CommandEncoder),
        #[cfg(test)] force_separate: bool,
    ) -> Result<()> {
        #[cfg(test)]
        if let Some(trace) = &self.newton_work_trace {
            trace.validate_next()?;
        }
        #[cfg(test)]
        if let Some(trace) = &self.admm_work_trace {
            trace.validate_next()?;
        }
        let before = |encoder: &mut wgpu::CommandEncoder| {
            if let Some(jacobians) = &self.shared_jacobians {
                for &[source, destination, bytes] in &self.jacobian_copies {
                    encoder.copy_buffer_to_buffer(
                        &jacobians[current],
                        source,
                        &self.jacobian_buffer,
                        destination,
                        bytes,
                    );
                }
            }
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.guard_pipeline);
                pass.set_bind_group(0, &self.guard_groups[current], &[]);
                pass.dispatch_workgroups(1, 1, 1);
                molla_core::gpu_metrics::record_dispatch();
            }
        };
        let publish = |encoder: &mut wgpu::CommandEncoder| {
            self.publication[current].encode(encoder);
            #[cfg(test)]
            if let Some(trace) = &self.admm_work_trace {
                trace.encode(encoder, current);
            }
            after(encoder);
        };
        let composed = composed_submissions(
            #[cfg(test)]
            force_separate,
        );
        if composed {
            self.coupling[current].step_with_commands(soil, dt, before, publish)?;
            #[cfg(test)]
            {
                self.submission_counts[0] += 1;
                self.submission_counts[1] += 1;
            }
        } else {
            let device = _engine.device().unwrap();
            let queue = _engine.queue().unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            before(&mut encoder);
            queue.submit([encoder.finish()]);
            self.coupling[current].step(soil, dt)?;
            let mut encoder = device.create_command_encoder(&Default::default());
            publish(&mut encoder);
            queue.submit([encoder.finish()]);
            #[cfg(test)]
            {
                self.submission_counts[0] += 1;
                self.submission_counts[1] += 3;
            }
        }
        #[cfg(test)]
        if let Some(trace) = &mut self.newton_work_trace {
            trace.encode(
                _engine,
                self.coupling[current].status_buffer(),
                self.coupling[current].newton_failure_buffer().unwrap(),
            );
        }
        self.last_input = Some(current);
        Ok(())
    }

    pub(super) fn read_contact_motion(
        &self,
        engine: &Engine,
        timeout: std::time::Duration,
    ) -> Result<Vec<crate::soil_contact_motion_gpu::SoilContactMotion>> {
        let index = self
            .last_input
            .ok_or_else(|| invalid("no projected contact sample"))?;
        let reader = match &self.coupling[index] {
            Coupling::Unilateral(p) => p.contact_motion(),
            Coupling::Sticking(_) => {
                return Err(invalid("contact motion requires unilateral coupling"));
            }
        };
        engine.check_device()?;
        let device = engine.device().unwrap();
        let completion =
            soil_completion::CompletionDeadline::new(engine.device_health().unwrap(), timeout)?;
        let mut encoder = device.create_command_encoder(&Default::default());
        reader.encode(&mut encoder);
        engine.queue().unwrap().submit([encoder.finish()]);
        let result = (|| {
            map_read_with_completion(device, reader.staging(), Some(&completion))?;
            let data = molla_core::gpu_readback!(reader.staging().slice(..));
            reader.decode(&data)
        })();
        reader.staging().unmap();
        result
    }
}
