use super::*;
use crate::{GpuGmres, GpuGmresSettings};
use std::num::NonZeroU64;

#[derive(Clone, Copy)]
pub struct CoupledNewtonBindings<'a> {
    pub soft_velocity: &'a wgpu::Buffer,
    pub rigid_velocity: &'a wgpu::Buffer,
    pub impulses: &'a wgpu::Buffer,
    pub soft_direction: &'a wgpu::Buffer,
    pub rigid_direction: &'a wgpu::Buffer,
    pub impulse_direction: &'a wgpu::Buffer,
    pub material_residual: &'a wgpu::Buffer,
    pub material_action: &'a wgpu::Buffer,
    pub contact: &'a ContactLinearizationKernel,
}

#[derive(Clone, Copy)]
pub struct CoupledNewtonSettings {
    pub iterations: u32,
    pub backtracks: u32,
    pub absolute_tolerance: f32,
    pub relative_tolerance: f32,
    pub armijo: f32,
    pub linear: GpuGmresSettings,
}
impl CoupledNewtonSettings {
    pub fn validate(self) -> Result<()> {
        self.linear.validate()?;
        if !(1..=64).contains(&self.iterations)
            || !(1..=24).contains(&self.backtracks)
            || !self.absolute_tolerance.is_finite()
            || self.absolute_tolerance < 0.0
            || !self.relative_tolerance.is_finite()
            || !(0.0..1.0).contains(&self.relative_tolerance)
            || (self.absolute_tolerance == 0.0 && self.relative_tolerance == 0.0)
            || !self.armijo.is_finite()
            || !(0.0..1.0).contains(&self.armijo)
            || self.armijo == 0.0
        {
            return Err(Error::Build("invalid coupled Newton settings".into()));
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CoupledNewtonDiagnostics {
    /// Bits: 0 invalid initial residual; 1 linear arithmetic fault; 2 rejected search.
    pub faults: u32,
    pub running: u32,
    pub accepted: u32,
    pub iterations: u32,
    pub initial_norm: f32,
    pub residual_norm: f32,
    pub base_norm: f32,
    pub threshold: f32,
    pub accepted_alpha: f32,
    pub trial_norm: f32,
    pub converged: u32,
    pub line_evaluations: u32,
    pub linear_iterations: u32,
    pub linear_residual_norm: f32,
    pub linear_converged: u32,
    pub linear_faults: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    particles: u32,
    rows: u32,
    dofs: u32,
    dimension: u32,
    alpha: f32,
    absolute_tolerance: f32,
    relative_tolerance: f32,
    armijo: f32,
}

/// Fixed-geometry simultaneous material/contact Newton solve on scratch candidate buffers.
/// The material target includes the contact reference impulses; rigid reference and
/// prepared contact multipliers remain frozen during the entire solve.
pub struct CoupledNewtonKernel {
    particles: u32,
    rows: u32,
    dofs: u32,
    dimension: u32,
    layout: wgpu::BindGroupLayout,
    pipelines: [wgpu::ComputePipeline; 14],
    gmres: GpuGmres,
    unity: wgpu::Buffer,
    inputs: [wgpu::Buffer; 12],
    packed: wgpu::Buffer,
    rhs: wgpu::Buffer,
    base: wgpu::Buffer,
    partials: wgpu::Buffer,
    diagnostics: wgpu::Buffer,
    history: wgpu::Buffer,
}

impl CoupledNewtonKernel {
    pub fn new(
        device: &wgpu::Device,
        particles: u32,
        rows: u32,
        dofs: u32,
        restart: u32,
        bindings: CoupledNewtonBindings<'_>,
    ) -> Result<Self> {
        let limits = device.limits();
        let dimension = 3 * u64::from(particles) + u64::from(dofs) + 3 * u64::from(rows);
        if particles == 0
            || rows == 0
            || !(1..=64).contains(&dofs)
            || dimension > u64::from(u32::MAX)
            || dimension * 4 > limits.max_buffer_size
            || dimension * 4 > u64::from(limits.max_storage_buffer_binding_size)
            || dimension.div_ceil(64) > u64::from(limits.max_compute_workgroups_per_dimension)
            || limits.max_storage_buffers_per_shader_stage < 20
            || limits.max_bindings_per_bind_group < 21
            || bindings.contact.particles != particles
            || bindings.contact.rows != rows
            || bindings.contact.dofs != dofs
        {
            return Err(Error::Build(
                "invalid coupled Newton dimensions or device limits".into(),
            ));
        }
        let inputs = [
            bindings.soft_velocity,
            bindings.rigid_velocity,
            bindings.impulses,
            bindings.soft_direction,
            bindings.rigid_direction,
            bindings.impulse_direction,
            bindings.material_residual,
            bindings.material_action,
            bindings.contact.soft_terms(),
            bindings.contact.rigid_terms(),
            bindings.contact.contact_terms(),
            bindings.contact.status(),
        ];
        let minimum = [
            u64::from(particles) * 16,
            u64::from(dofs) * 4,
            u64::from(rows) * 16,
            u64::from(particles) * 16,
            u64::from(dofs) * 4,
            u64::from(rows) * 16,
            u64::from(particles) * 16,
            u64::from(particles) * 16,
            u64::from(particles) * 32,
            u64::from(dofs) * 16,
            u64::from(rows) * 32,
            8,
        ];
        for (i, (buffer, size)) in inputs.iter().zip(minimum).enumerate() {
            if buffer.size() < size
                || buffer.size() > u64::from(limits.max_storage_buffer_binding_size)
                || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
                || inputs
                    .iter()
                    .enumerate()
                    .any(|(j, other)| i != j && (i < 6 || j < 6) && buffer == other)
            {
                return Err(Error::Build(
                    "invalid or aliased coupled Newton buffer".into(),
                ));
            }
        }
        let gmres = GpuGmres::new(device, dimension as u32, restart)?;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("coupled FEM Newton"),
            entries: &(0..21)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 19 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: (6..14).contains(&binding) || binding == 20,
                            }
                        },
                        has_dynamic_offset: binding == 19,
                        min_binding_size: if binding == 19 {
                            NonZeroU64::new(32)
                        } else {
                            None
                        },
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("coupled FEM Newton"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/fem_coupled_newton.wgsl").into(),
            ),
        });
        let pipelines = [
            "zero_directions",
            "unpack_direction",
            "pack_rhs",
            "pack_action",
            "norm_partials",
            "initial_norm",
            "begin_iteration",
            "capture_base",
            "check_linear",
            "apply_trial",
            "accept_trial",
            "finish_iteration",
            "restore_rejected",
            "finish_solve",
        ]
        .map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
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
            particles,
            rows,
            dofs,
            dimension: dimension as u32,
            layout,
            pipelines,
            gmres,
            unity: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Newton unscaled residual weights"),
                contents: bytemuck::cast_slice(&vec![1.0_f32; dimension as usize]),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            inputs: inputs.map(Clone::clone),
            packed: allocate("Newton packed vector", dimension * 4),
            rhs: allocate("Newton right hand side", dimension * 4),
            base: allocate("Newton accepted base", dimension * 4),
            partials: allocate("Newton residual norm partials", dimension.div_ceil(64) * 8),
            diagnostics: allocate("Newton diagnostics", 64),
            history: allocate("Newton iteration history", 64 * 64),
        })
    }

    /// Encodes right-preconditioned Newton/Krylov/backtracking iterations. `evaluate` must refresh
    /// both material and contact residual/actions at the bound candidates/directions.
    /// Candidate buffers contain the last accepted state on return, not an unaccepted trial.
    /// Every linear correction starts from zero, regardless of the linear initial-guess setting.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        inverse_diagonal: &wgpu::Buffer,
        settings: CoupledNewtonSettings,
        mut evaluate: impl FnMut(&mut wgpu::CommandEncoder) -> Result<()>,
    ) -> Result<()> {
        settings.validate()?;
        if inverse_diagonal.size() != u64::from(self.dimension) * 4
            || !inverse_diagonal
                .usage()
                .contains(wgpu::BufferUsages::STORAGE)
            || self
                .inputs
                .iter()
                .chain([
                    &self.packed,
                    &self.rhs,
                    &self.base,
                    &self.partials,
                    &self.diagnostics,
                    &self.history,
                    &self.unity,
                    self.gmres.solution(),
                    self.gmres.diagnostics(),
                ])
                .any(|b| b == inverse_diagonal)
        {
            return Err(Error::Build(
                "invalid coupled Newton settings/preconditioner".into(),
            ));
        }
        self.gmres
            .validate_inputs(&self.rhs, inverse_diagonal, settings.linear)?;
        let stride = device.limits().min_uniform_buffer_offset_alignment.max(32);
        let mut bytes = vec![0; (stride * settings.backtracks) as usize];
        for slot in 0..settings.backtracks {
            let params = Params {
                particles: self.particles,
                rows: self.rows,
                dofs: self.dofs,
                dimension: self.dimension,
                alpha: 0.5_f32.powi(slot as i32),
                absolute_tolerance: settings.absolute_tolerance,
                relative_tolerance: settings.relative_tolerance,
                armijo: settings.armijo,
            };
            bytes[(slot * stride) as usize..(slot * stride + 32) as usize]
                .copy_from_slice(bytemuck::bytes_of(&params));
        }
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut entries: Vec<_> = self
            .inputs
            .iter()
            .chain([
                self.gmres.solution(),
                self.gmres.diagnostics(),
                &self.packed,
                &self.rhs,
                &self.base,
                &self.partials,
                &self.diagnostics,
            ])
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: 19,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &uniform,
                offset: 0,
                size: NonZeroU64::new(32),
            }),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 20,
            resource: inverse_diagonal.as_entire_binding(),
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.layout,
            entries: &entries,
        });
        let dispatch =
            |encoder: &mut wgpu::CommandEncoder, pipeline: usize, slot: u32, count: u32| {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipelines[pipeline]);
                pass.set_bind_group(0, &group, &[slot * stride]);
                pass.dispatch_workgroups(count, 1, 1);
            };
        let vectors = self.particles.max(self.rows).max(self.dofs).div_ceil(64);
        let blocks = self.dimension.div_ceil(64);
        encoder.clear_buffer(&self.diagnostics, 0, None);
        encoder.clear_buffer(&self.history, 0, None);
        dispatch(encoder, 0, 0, vectors);
        evaluate(encoder)?;
        dispatch(encoder, 2, 0, blocks);
        dispatch(encoder, 4, 0, blocks);
        dispatch(encoder, 5, 0, 1);
        for iteration in 0..settings.iterations {
            dispatch(encoder, 6, 0, 1);
            dispatch(encoder, 7, 0, blocks);
            self.gmres.encode(
                device,
                encoder,
                &self.rhs,
                &self.unity,
                GpuGmresSettings {
                    zero_initial_guess: true,
                    ..settings.linear
                },
                |encoder, input, output| {
                    encoder.copy_buffer_to_buffer(
                        input,
                        0,
                        &self.packed,
                        0,
                        u64::from(self.dimension) * 4,
                    );
                    dispatch(encoder, 1, 0, vectors);
                    evaluate(encoder)?;
                    dispatch(encoder, 3, 0, blocks);
                    encoder.copy_buffer_to_buffer(
                        &self.packed,
                        0,
                        output,
                        0,
                        u64::from(self.dimension) * 4,
                    );
                    Ok(())
                },
            )?;
            dispatch(encoder, 8, 0, 1);
            for slot in 0..settings.backtracks {
                dispatch(encoder, 9, slot, vectors);
                evaluate(encoder)?;
                dispatch(encoder, 2, slot, blocks);
                dispatch(encoder, 4, slot, blocks);
                dispatch(encoder, 10, slot, 1);
            }
            dispatch(encoder, 11, 0, 1);
            dispatch(encoder, 12, 0, vectors);
            encoder.copy_buffer_to_buffer(
                &self.diagnostics,
                0,
                &self.history,
                u64::from(iteration) * 64,
                64,
            );
        }
        dispatch(encoder, 0, 0, vectors);
        evaluate(encoder)?;
        dispatch(encoder, 2, 0, blocks);
        dispatch(encoder, 4, 0, blocks);
        dispatch(encoder, 13, 0, 1);
        Ok(())
    }

    pub fn diagnostics(&self) -> &wgpu::Buffer {
        &self.diagnostics
    }
    /// Up to 64 iteration diagnostics, cleared at solve start and recorded before final evaluation.
    pub fn iteration_history(&self) -> &wgpu::Buffer {
        &self.history
    }
    /// Negative physical residual at the final accepted candidates, in mixed scalar order.
    pub fn residual_rhs(&self) -> &wgpu::Buffer {
        &self.rhs
    }
}
