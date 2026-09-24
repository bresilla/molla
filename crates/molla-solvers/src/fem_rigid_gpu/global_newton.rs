use super::*;
use molla_kernels::soft_rigid_solve::{
    ContactLinearizationBindings, ContactLinearizationKernel, CoupledNewtonBindings,
    CoupledNewtonKernel, CoupledNewtonSettings, GlobalContactBridge, GlobalContactCommitBindings,
};

#[derive(Clone, Copy)]
pub struct GlobalNewtonConfig {
    /// Maximum active rows; overflow rejects the substep rather than dropping constraints.
    pub contact_capacity: u32,
    pub restart: u32,
    pub solve: CoupledNewtonSettings,
}
impl Default for GlobalNewtonConfig {
    fn default() -> Self {
        Self {
            contact_capacity: 16384,
            restart: 32,
            solve: CoupledNewtonSettings {
                iterations: 12,
                backtracks: 8,
                absolute_tolerance: 1e-4,
                relative_tolerance: 0.0,
                armijo: 1e-4,
                block_maximum: true,
                linear: molla_kernels::GpuGmresSettings {
                    cycles: 4,
                    absolute_tolerance: 1e-7,
                    relative_tolerance: 1e-4,
                    zero_initial_guess: true,
                },
            },
        }
    }
}

pub(super) struct GlobalNewtonState {
    config: GlobalNewtonConfig,
    dofs: u32,
    candidate: State,
    rigid_candidate: wgpu::Buffer,
    rigid_reference: wgpu::Buffer,
    soft_direction: wgpu::Buffer,
    rigid_direction: wgpu::Buffer,
    impulse_direction: wgpu::Buffer,
    bridge: GlobalContactBridge,
    contact: ContactLinearizationKernel,
    newton: Option<CoupledNewtonKernel>,
    material_outputs: Option<[wgpu::Buffer; 2]>,
}

impl GlobalNewtonState {
    pub(super) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        bodies: usize,
        dofs: usize,
        rows: u32,
        mut config: GlobalNewtonConfig,
    ) -> Result<Self> {
        config.solve.validate()?;
        config.contact_capacity = config.contact_capacity.min(rows);
        let n = model.particle_count as u32;
        let r = config.contact_capacity;
        let d = dofs as u32;
        let dimension = 3 * u64::from(n) + 3 * u64::from(r) + u64::from(d);
        if r == 0
            || !(1..=64).contains(&config.restart)
            || u64::from(config.restart) > dimension
            || model.linear_damping != 0.0
            || !model.vertex_cable_bending_entries.is_empty()
            || device.limits().max_bind_groups < 7
            || device.limits().max_storage_buffers_per_shader_stage < 47
            || dimension * (u64::from(config.restart) + 1) * 4
                > u64::from(device.limits().max_storage_buffer_binding_size)
        {
            return Err(Error::Build(
                "unsupported native global Newton configuration".into(),
            ));
        }
        let mut candidate = model.state()?;
        candidate.upload_to_device(device, queue)?;
        let allocate = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                mapped_at_creation: false,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
        };
        Ok(Self {
            config,
            dofs: d,
            candidate,
            rigid_candidate: allocate("global rigid candidate", u64::from(d) * 4),
            rigid_reference: allocate("global rigid reference", u64::from(d) * 4),
            soft_direction: allocate("global soft direction", u64::from(n) * 16),
            rigid_direction: allocate("global rigid direction", u64::from(d) * 4),
            impulse_direction: allocate("global contact direction", u64::from(r) * 16),
            bridge: GlobalContactBridge::new(device, rows, r, n, d)?,
            contact: ContactLinearizationKernel::new(device, n, r, d, bodies as u32)?,
            newton: None,
            material_outputs: None,
        })
    }
    pub(super) fn diagnostics(&self) -> Option<&wgpu::Buffer> {
        self.newton.as_ref().map(CoupledNewtonKernel::diagnostics)
    }
    pub(super) fn iteration_history(&self) -> Option<&wgpu::Buffer> {
        self.newton
            .as_ref()
            .map(CoupledNewtonKernel::iteration_history)
    }
    pub(super) fn residual_rhs(&self) -> Option<&wgpu::Buffer> {
        self.newton.as_ref().map(CoupledNewtonKernel::residual_rhs)
    }
    pub(super) fn commit_status(&self) -> &wgpu::Buffer {
        self.bridge.commit_status()
    }

    pub(super) fn run(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        fem: &mut VbdSolver,
        model: &Model,
        input: &mut State,
        predicted: &State,
        source: &SoftRigidSolveKernel,
        original: SoftRigidSolveBindings<'_>,
        settings: SoftRigidSolveSettings,
        dt: Real,
    ) -> Result<()> {
        source.encode_preparation_with_schedule(device, encoder, original, settings)?;
        self.bridge
            .encode_prepare(device, encoder, source, original.contacts)?;
        macro_rules! copy {
            ($field:ident) => {{
                let from = predicted.$field.device_buffer().unwrap();
                let to = self.candidate.$field.device_buffer().unwrap();
                encoder.copy_buffer_to_buffer(from, 0, to, 0, from.size());
                self.candidate.$field.mark_device_dirty();
            }};
        }
        copy!(particle_q);
        copy!(particle_qd);
        copy!(tet_f_plastic);
        copy!(tet_plastic_strain);
        copy!(tet_back_stress);
        encoder.copy_buffer_to_buffer(
            original.rigid_velocities,
            0,
            &self.rigid_reference,
            0,
            self.rigid_reference.size(),
        );
        encoder.copy_buffer_to_buffer(
            original.rigid_velocities,
            0,
            &self.rigid_candidate,
            0,
            self.rigid_candidate.size(),
        );
        self.contact.prepare(
            device,
            encoder,
            self.bridge.compact_solve(),
            ContactLinearizationBindings {
                state: SoftRigidSolveBindings {
                    velocities: self.candidate.particle_qd.device_buffer().unwrap(),
                    rigid_velocities: &self.rigid_candidate,
                    contacts: self.bridge.contacts(),
                    ..original
                },
                rigid_reference: &self.rigid_reference,
                soft_direction: &self.soft_direction,
                rigid_direction: &self.rigid_direction,
                impulses: self.bridge.impulses(),
                impulse_direction: &self.impulse_direction,
            },
            settings,
        )?;
        let pending =
            std::mem::replace(encoder, device.create_command_encoder(&Default::default()));
        queue.submit([pending.finish()]);
        let (residual, action) = fem.material_linearization_gpu(
            model,
            input,
            &mut self.candidate,
            dt,
            &self.soft_direction,
        )?;
        let outputs = [residual.clone(), action.clone()];
        fem.encode_material_inverse_diagonal_gpu(encoder)?;
        self.bridge
            .encode_material_scale(device, encoder, &outputs[1])?;
        if self.material_outputs.as_ref() != Some(&outputs) {
            self.newton = Some(CoupledNewtonKernel::new(
                device,
                model.particle_count as u32,
                self.config.contact_capacity,
                self.dofs,
                self.config.restart,
                CoupledNewtonBindings {
                    soft_velocity: self.candidate.particle_qd.device_buffer().unwrap(),
                    rigid_velocity: &self.rigid_candidate,
                    impulses: self.bridge.impulses(),
                    soft_direction: &self.soft_direction,
                    rigid_direction: &self.rigid_direction,
                    impulse_direction: &self.impulse_direction,
                    material_residual: &outputs[0],
                    material_action: &outputs[1],
                    contact: &self.contact,
                },
            )?);
            self.material_outputs = Some(outputs);
        }
        let newton = self.newton.as_ref().unwrap();
        let mut evaluations = 0_u32;
        newton.encode_split_with_finalize(
            device,
            encoder,
            self.bridge.inverse_diagonal(),
            self.config.solve,
            |encoder, action_only| {
                if action_only {
                    fem.encode_cached_material_action_gpu(encoder)?;
                } else {
                    fem.encode_material_linearization_gpu(encoder)?;
                    fem.encode_material_tangent_cache_gpu(encoder)?;
                }
                self.contact.encode(encoder)?;
                evaluations += 1;
                if evaluations % 16 == 0 {
                    let pending = std::mem::replace(
                        encoder,
                        device.create_command_encoder(&Default::default()),
                    );
                    queue.submit([pending.finish()]);
                }
                Ok(())
            },
            |encoder| {
                self.bridge.encode_project(encoder);
                Ok(())
            },
        )?;
        self.bridge.encode_commit(
            device,
            encoder,
            source,
            GlobalContactCommitBindings {
                soft_candidate: self.candidate.particle_qd.device_buffer().unwrap(),
                rigid_candidate: &self.rigid_candidate,
                soft_terms: self.contact.soft_terms(),
                newton_diagnostics: newton.diagnostics(),
                soft_velocity: original.velocities,
                rigid_velocity: original.rigid_velocities,
                input_velocity: input.particle_qd.device_buffer().unwrap(),
            },
        )?;
        input.particle_qd.mark_device_dirty();
        Ok(())
    }
}
