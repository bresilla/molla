use molla_core::{Error, Result};
use std::num::NonZeroU64;
use wgpu::util::DeviceExt;

#[derive(Clone, Copy)]
pub struct GpuGmresSettings {
    pub cycles: u32,
    pub absolute_tolerance: f32,
    pub relative_tolerance: f32,
    pub zero_initial_guess: bool,
}
impl GpuGmresSettings {
    pub fn validate(self) -> Result<()> {
        if self.cycles == 0
            || self.cycles > 256
            || !self.absolute_tolerance.is_finite()
            || self.absolute_tolerance < 0.0
            || !self.relative_tolerance.is_finite()
            || !(0.0..1.0).contains(&self.relative_tolerance)
            || (self.absolute_tolerance == 0.0 && self.relative_tolerance == 0.0)
        {
            return Err(Error::Build("invalid GPU GMRES settings".into()));
        }
        Ok(())
    }
}

/// Device diagnostics; convergence is checked against the unpreconditioned residual.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuGmresDiagnostics {
    /// Bit 0: nonfinite arithmetic/input. Bit 2: invalid inverse preconditioner.
    pub faults: u32,
    pub active: u32,
    pub used: u32,
    pub iterations: u32,
    pub initial_norm: f32,
    pub residual_norm: f32,
    pub estimated_norm: f32,
    pub inner_threshold: f32,
    pub rhs_norm: f32,
    pub threshold: f32,
    pub converged: u32,
    pub breakdowns: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    restart: u32,
    column: u32,
    mode: u32,
    absolute_tolerance: f32,
    relative_tolerance: f32,
    pad: [u32; 2],
}

/// Restarted left-preconditioned GMRES with two-pass classical Gram-Schmidt on GPU.
/// Host code encodes a bounded dispatch schedule; vectors, reductions, rotations,
/// backsolve, stopping decisions and true residual checks remain device-resident.
pub struct GpuGmres {
    n: u32,
    restart: u32,
    layout: wgpu::BindGroupLayout,
    pipelines: [wgpu::ComputePipeline; 13],
    solution: wgpu::Buffer,
    direction: wgpu::Buffer,
    work: wgpu::Buffer,
    basis: wgpu::Buffer,
    small: wgpu::Buffer,
    partials: wgpu::Buffer,
    diagnostics: wgpu::Buffer,
}

impl GpuGmres {
    pub fn new(device: &wgpu::Device, n: u32, restart: u32) -> Result<Self> {
        if n == 0 || !(1..=64).contains(&restart) || restart > n {
            return Err(Error::Build("invalid GPU GMRES dimensions".into()));
        }
        let limits = device.limits();
        let sizes = [
            u64::from(n) * 4,
            u64::from(n) * (u64::from(restart) + 1) * 4,
            (u64::from(restart) * (u64::from(restart) + 1) + 5 * u64::from(restart) + 1) * 4,
            u64::from(n.div_ceil(64)) * u64::from(restart) * 16,
        ];
        if limits.max_storage_buffers_per_shader_stage < 9
            || limits.max_bindings_per_bind_group < 10
            || limits.max_dynamic_uniform_buffers_per_pipeline_layout < 1
            || limits.max_uniform_buffer_binding_size < 32
            || limits.max_compute_invocations_per_workgroup < 64
            || limits.max_compute_workgroup_size_x < 64
            || limits.max_compute_workgroup_storage_size < 1024
            || n.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || restart > limits.max_compute_workgroups_per_dimension
            || u64::from(n) * (u64::from(restart) + 1) > u64::from(u32::MAX)
            || sizes
                .iter()
                .any(|&s| s > limits.max_buffer_size || s > limits.max_storage_buffer_binding_size)
        {
            return Err(Error::Build(
                "invalid GPU GMRES dimensions or device limits".into(),
            ));
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GPU GMRES"),
            entries: &(0..10)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 9 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: binding < 2,
                            }
                        },
                        has_dynamic_offset: binding == 9,
                        min_binding_size: if binding == 9 {
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
            label: Some("GPU GMRES"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GPU GMRES"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/gmres.wgsl").into()),
        });
        let pipelines = [
            "seed",
            "residual",
            "residual_reduce",
            "initial_basis",
            "precondition",
            "dot_partials",
            "dot_reduce",
            "subtract",
            "norm_partials",
            "arnoldi_reduce",
            "next_basis",
            "backsolve",
            "update_solution",
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
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    mapped_at_creation: false,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                }
            )
        };
        Ok(Self {
            n,
            restart,
            layout,
            pipelines,
            solution: allocate("GMRES solution", sizes[0]),
            direction: allocate("GMRES operator input", sizes[0]),
            work: allocate("GMRES operator output", sizes[0]),
            basis: allocate("GMRES Krylov basis", sizes[1]),
            small: allocate("GMRES Hessenberg and rotations", sizes[2]),
            partials: allocate("GMRES reduction partials", sizes[3]),
            diagnostics: allocate("GMRES diagnostics", 48),
        })
    }

    pub(crate) fn validate_inputs(
        &self,
        rhs: &wgpu::Buffer,
        inverse_diagonal: &wgpu::Buffer,
        settings: GpuGmresSettings,
    ) -> Result<()> {
        let owned = [
            &self.solution,
            &self.direction,
            &self.work,
            &self.basis,
            &self.small,
            &self.partials,
            &self.diagnostics,
        ];
        settings.validate()?;
        if [rhs, inverse_diagonal].iter().any(|b| {
            b.size() != u64::from(self.n) * 4
                || !b.usage().contains(wgpu::BufferUsages::STORAGE)
                || owned.contains(b)
        }) {
            return Err(Error::Build(
                "invalid GPU GMRES settings or aliased input".into(),
            ));
        }
        Ok(())
    }

    /// Encodes `A*x=b`; the callback must overwrite output with the linear operator
    /// applied to input without changing input, RHS, preconditioner or physical state.
    /// The initial guess is read from `solution()` unless explicitly zeroed.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        rhs: &wgpu::Buffer,
        inverse_diagonal: &wgpu::Buffer,
        settings: GpuGmresSettings,
        mut operator: impl FnMut(&mut wgpu::CommandEncoder, &wgpu::Buffer, &wgpu::Buffer) -> Result<()>,
    ) -> Result<()> {
        self.validate_inputs(rhs, inverse_diagonal, settings)?;
        let owned = [
            &self.solution,
            &self.direction,
            &self.work,
            &self.basis,
            &self.small,
            &self.partials,
            &self.diagnostics,
        ];
        let stride = device.limits().min_uniform_buffer_offset_alignment.max(32);
        let mut bytes = vec![0; stride as usize * (2 + 2 * self.restart) as usize];
        for slot in 0..2 + 2 * self.restart {
            let params = Params {
                n: self.n,
                restart: self.restart,
                column: if slot < 2 { 0 } else { (slot - 2) / 2 },
                mode: slot % 2,
                absolute_tolerance: settings.absolute_tolerance,
                relative_tolerance: settings.relative_tolerance,
                pad: [0; 2],
            };
            bytes[(slot * stride) as usize..(slot * stride + 32) as usize]
                .copy_from_slice(bytemuck::bytes_of(&params));
        }
        let uniforms = molla_core::gpu_buffer_init!(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("GMRES dispatch parameters"),
                contents: &bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            }
        );
        let mut entries: Vec<_> = [rhs, inverse_diagonal]
            .into_iter()
            .chain(owned)
            .enumerate()
            .map(|(binding, b)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: 9,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &uniforms,
                offset: 0,
                size: NonZeroU64::new(32),
            }),
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GPU GMRES"),
            layout: &self.layout,
            entries: &entries,
        });
        let dispatch = |pass: &mut wgpu::ComputePass<'_>, pipeline: usize, slot: u32, x, y| {
            pass.set_pipeline(&self.pipelines[pipeline]);
            pass.set_bind_group(0, &group, &[slot * stride]);
            pass.dispatch_workgroups(x, y, 1);
        };
        encoder.clear_buffer(&self.diagnostics, 0, None);
        if settings.zero_initial_guess {
            encoder.clear_buffer(&self.solution, 0, None);
        }
        let blocks = self.n.div_ceil(64);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            dispatch(&mut pass, 0, 0, blocks, 1);
        }
        operator(encoder, &self.direction, &self.work)?;
        for cycle in 0..=settings.cycles {
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                dispatch(&mut pass, 1, 0, blocks, 1);
                dispatch(&mut pass, 2, u32::from(cycle != 0), 1, 1);
                if cycle < settings.cycles {
                    dispatch(&mut pass, 3, 0, blocks, 1);
                }
            }
            if cycle == settings.cycles {
                break;
            }
            for k in 0..self.restart {
                operator(encoder, &self.direction, &self.work)?;
                let slot = 2 + 2 * k;
                let mut pass = encoder.begin_compute_pass(&Default::default());
                dispatch(&mut pass, 4, slot, blocks, 1);
                for reorthogonalization in 0..2 {
                    dispatch(&mut pass, 5, slot + reorthogonalization, blocks, k + 1);
                    dispatch(&mut pass, 6, slot + reorthogonalization, k + 1, 1);
                    dispatch(&mut pass, 7, slot + reorthogonalization, blocks, 1);
                }
                dispatch(&mut pass, 8, slot, blocks, 1);
                dispatch(&mut pass, 9, slot, 1, 1);
                dispatch(&mut pass, 10, slot, blocks, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                dispatch(&mut pass, 11, 0, 1, 1);
                dispatch(&mut pass, 12, 0, blocks, 1);
                dispatch(&mut pass, 0, 0, blocks, 1);
            }
            operator(encoder, &self.direction, &self.work)?;
        }
        Ok(())
    }

    pub fn solution(&self) -> &wgpu::Buffer {
        &self.solution
    }
    pub fn diagnostics(&self) -> &wgpu::Buffer {
        &self.diagnostics
    }
}
