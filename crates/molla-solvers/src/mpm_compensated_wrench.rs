use super::*;

#[cfg(all(test, feature = "resource-counters"))]
#[path = "mpm_compensated_wrench_tests.rs"]
mod tests;

fn invalid(message: &str) -> Error {
    Error::Solver(format!("Compensated wrench: {message}"))
}

fn replace(source: &mut String, old: &str, new: &str) -> Result<()> {
    if source.matches(old).count() != 1 {
        return Err(invalid("shader source contract changed"));
    }
    *source = source.replace(old, new);
    Ok(())
}

pub(super) fn source() -> Result<String> {
    let mut source = include_str!("../shaders/mpm.wgsl").to_owned();
    for entry in ["rigid_imprint", "mpm_rigid_force_integrate"] {
        let signature = format!("fn {entry}(@builtin(global_invocation_id) gid: vec3<u32>) {{");
        replace(
            &mut source,
            &signature,
            &format!("{signature}\n    wrench_lane=gid.x%64u;"),
        )?;
    }
    let signature = "fn atomic_add_rigid_wrench(idx: u32, val: f32) {";
    let start = source
        .find(signature)
        .ok_or_else(|| invalid("missing wrench helper"))?;
    let end = start
        + source[start..]
            .find("\n}\n")
            .ok_or_else(|| invalid("unterminated wrench helper"))?
        + 2;
    source.replace_range(start..end,"fn atomic_add_rigid_wrench(idx: u32, val: f32) {\n    var old_bits=atomicLoad(&rigid_wrench[idx]);\n    loop {\n        let old=bitcast<f32>(old_bits);let next=wrench_round(old+val);\n        let result=atomicCompareExchangeWeak(&rigid_wrench[idx],old_bits,bitcast<u32>(next));\n        if result.exchanged {wrench_add_tail(idx,wrench_error(old,val,next));break;}\n        old_bits=result.old_value;\n    }\n}");
    replace(
        &mut source,
        "        atomicStore(&rigid_wrench[base + k], bitcast<GridAtomicScalar>(0u));",
        "        atomicStore(&rigid_wrench[base + k], bitcast<GridAtomicScalar>(0u));\n        atomicStore(&wrench_tail[base+k],0u);",
    )?;
    source.push_str(include_str!("../shaders/mpm_compensated_wrench.wgsl"));
    Ok(source)
}

pub(super) struct Reduction {
    tail: wgpu::Buffer,
    bindings: wgpu::BindGroup,
    reset: wgpu::ComputePipeline,
    imprint: [wgpu::ComputePipeline; 2],
    shell: [wgpu::ComputePipeline; 2],
    resolve: wgpu::ComputePipeline,
    enabled: bool,
}

impl Reduction {
    fn new(sim: &MpmSimulation) -> Result<Self> {
        let device = &sim.device;
        let bytes = sim.rigid_wrench.size();
        let limits = device.limits();
        if bytes > limits.max_storage_buffer_binding_size
            || bytes > limits.max_buffer_size
            || limits.max_storage_buffers_per_shader_stage < 19
            || limits.max_compute_workgroup_storage_size < 256
        {
            return Err(invalid("device resource limits exceeded"));
        }
        let source = source()?;
        let tail = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("compensated rigid wrench tail"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false
            }
        );
        let rigid = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &mixed_storage_entries(&[true, true, false, false]),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[
                Some(&sim.layout_uniforms),
                Some(&sim.layout_particles),
                Some(&sim.layout_grid),
                Some(&rigid),
            ],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("compensated rigid wrench"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = |entry, active| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[
                        ("SPARSE_GRID", f64::from(sim.sparse_grid)),
                        ("ACTIVE_GRID", f64::from(active)),
                    ],
                    ..Default::default()
                },
                cache: None,
            })
        };
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &rigid,
            entries: &[
                buffer_entry(0, &sim.rigid_center),
                buffer_entry(1, &sim.rigid_vel),
                buffer_entry(2, &sim.rigid_wrench),
                buffer_entry(3, &tail),
            ],
        });
        Ok(Self {
            tail,
            bindings,
            reset: pipeline("mpm_rigid_wrench_reset", false),
            imprint: [false, true].map(|a| pipeline("rigid_imprint", a)),
            shell: [false, true].map(|a| pipeline("mpm_rigid_force_integrate", a)),
            resolve: pipeline("wrench_resolve", false),
            enabled: false,
        })
    }

    fn resolve(&self, pass: &mut wgpu::ComputePass<'_>, sim: &MpmSimulation) {
        pass.set_pipeline(&self.resolve);
        pass.dispatch_workgroups(sim.rigid_count.div_ceil(WORKGROUP_SIZE), 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }

    pub(super) fn shell(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        sim: &MpmSimulation,
        active: bool,
    ) {
        pass.set_bind_group(3, &self.bindings, &[]);
        dispatch_grid(pass, sim, &self.shell[usize::from(active)], active);
        self.resolve(pass, sim);
        pass.set_bind_group(3, &sim.bg_rigid, &[]);
    }
}

fn dispatch_grid(
    pass: &mut wgpu::ComputePass<'_>,
    sim: &MpmSimulation,
    pipeline: &wgpu::ComputePipeline,
    active: bool,
) {
    pass.set_pipeline(pipeline);
    if active {
        pass.dispatch_workgroups_indirect(&sim.active_tiles.indirect, 0);
    } else {
        pass.dispatch_workgroups(
            sim.cfg.grid_size.pow(3).div_ceil(WORKGROUP_SIZE).max(1),
            1,
            1,
        );
    }
    molla_core::gpu_metrics::record_dispatch();
}

impl MpmSimulation {
    /// Prepares non-native compensated reaction reduction, 32 * max(max_rigid, 1) bytes.
    /// Preparation does not enable it; no force terms or grid snapshots are retained.
    pub fn prepare_compensated_rigid_wrench(&mut self, max_bytes: u64) -> Result<()> {
        if self.native_float_atomics {
            return Err(invalid("native atomics are unsupported"));
        }
        if self.rigid_wrench.size() > max_bytes {
            return Err(invalid("tail-buffer budget exceeded"));
        }
        if self.compensated_wrench.is_none() {
            self.compensated_wrench = Some(Reduction::new(self)?);
            self.invalidate_prepared_steps();
        }
        Ok(())
    }

    /// Selects prepared compensated imprint/shell reduction and one final resolve.
    pub fn set_compensated_rigid_wrench(&mut self, enabled: bool) -> Result<()> {
        if enabled && (self.native_float_atomics || self.compensated_wrench.is_none()) {
            return Err(invalid("requires prepared non-native reduction"));
        }
        if let Some(reduction) = &mut self.compensated_wrench {
            reduction.enabled = enabled;
        }
        self.invalidate_prepared_steps();
        Ok(())
    }

    pub fn compensated_rigid_wrench_enabled(&self) -> bool {
        self.compensated_wrench.as_ref().is_some_and(|r| r.enabled)
    }
    pub fn compensated_rigid_wrench_storage_bytes(&self) -> u64 {
        self.compensated_wrench
            .as_ref()
            .map_or(0, |r| r.tail.size())
    }

    /// Releases disabled reaction-reduction storage and pipelines.
    pub fn release_compensated_rigid_wrench(&mut self) -> Result<()> {
        if self.compensated_rigid_wrench_enabled() {
            return Err(invalid("disable before release"));
        }
        if self.compensated_wrench.take().is_some() {
            self.invalidate_prepared_steps();
        }
        Ok(())
    }

    pub(super) fn rigid_wrench_reduction(&self) -> Option<&Reduction> {
        self.compensated_wrench.as_ref().filter(|r| r.enabled)
    }

    pub(super) fn encode_rigid_wrench_reset(&self, pass: &mut wgpu::ComputePass<'_>) {
        if let Some(reduction) = self.rigid_wrench_reduction() {
            pass.set_bind_group(3, &reduction.bindings, &[]);
            pass.set_pipeline(&reduction.reset);
        } else {
            pass.set_pipeline(&self.pipe_rigid_wrench_reset);
        }
        pass.dispatch_workgroups(self.rigid_count.div_ceil(WORKGROUP_SIZE), 1, 1);
        molla_core::gpu_metrics::record_dispatch();
        pass.set_bind_group(3, &self.bg_rigid, &[]);
    }

    pub(super) fn encode_rigid_imprint(&self, pass: &mut wgpu::ComputePass<'_>, active: bool) {
        if let Some(reduction) = self.rigid_wrench_reduction() {
            pass.set_bind_group(3, &reduction.bindings, &[]);
            dispatch_grid(pass, self, &reduction.imprint[usize::from(active)], active);
            if self.prescribed_solids {
                reduction.resolve(pass, self);
            }
            pass.set_bind_group(3, &self.bg_rigid, &[]);
        } else {
            dispatch_grid(
                pass,
                self,
                if active {
                    &self.active_pipelines[1]
                } else {
                    &self.pipe_rigid
                },
                active,
            );
        }
    }
}
