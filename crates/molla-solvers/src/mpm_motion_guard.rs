//! Per-substep displacement admission for swept soil residency.

use super::*;

#[path = "mpm_soil_quiescence.rs"]
mod quiescence;
pub use quiescence::{MpmSoilQuiescence, MpmSoilQuiescenceSnapshot};
pub use quiescence::{MpmSoilRetainedSupportParticle, MpmSoilRetainedSupportSnapshot};
pub use quiescence::{MpmSoilStencilSupport, MpmSoilSupportSnapshot};
pub use quiescence::{MpmSoilSupportClosureParticle, MpmSoilSupportClosureSnapshot};

#[path = "mpm_motion_fusion.rs"]
mod fusion;

const CHECK: &str = "    new_pos = clamped_pos;\n";
const PUBLICATION: &str = "    p_pos[pid] = vec4<f32>(new_pos, 0.0);\n    p_vel[pid] = vec4<f32>(v_new, 0.0);\n    p_C[pid] = C_new;\n    p_F[pid] = F_new;";

fn source() -> String {
    let source = include_str!("../shaders/mpm.wgsl");
    assert_eq!(source.matches(CHECK).count(), 1);
    let mut source = source.replace(CHECK, &format!("{CHECK}    if !soil_motion_admitted(pos, new_pos) {{\n        atomicOr(&g_kdiag[grid_metadata() + 2u], 4u);\n        return;\n    }}\n"));
    assert_eq!(source.matches(PUBLICATION).count(), 1);
    source = source.replace(PUBLICATION, "    soil_motion_proposals[pid] = SoilMotionProposal(new_pos, p_jp[pid], vec4<f32>(v_new, 0.0), C_new, F_new);");
    let (prefix, projection) = source
        .split_once("fn plasticity(")
        .expect("plasticity entry point");
    assert!(!projection.contains("\nfn ") && !projection.contains("@compute"));
    let projection = projection
        .replace("p_F[pid]", "soil_motion_proposals[pid].deformation")
        .replace("p_jp[pid]", "soil_motion_proposals[pid].plastic");
    source = format!("{prefix}fn soil_motion_plasticity({projection}");
    source.push_str(include_str!("../shaders/mpm_motion_guard.wgsl"));
    source.push_str(include_str!("../shaders/mpm_motion_commit.wgsl"));
    source
}

fn lower_bounds(bounds: [f64; 3]) -> [f32; 4] {
    let mut result = [0.0; 4];
    for (out, value) in result.iter_mut().zip(bounds) {
        let nearest = value.min(f64::from(f32::MAX)) as f32;
        *out = if f64::from(nearest) > value {
            nearest.next_down()
        } else {
            nearest
        };
        if *out < f32::MIN_POSITIVE {
            *out = 0.0;
        }
    }
    result
}

pub(super) struct MotionGuard {
    params: wgpu::Buffer,
    proposals: wgpu::Buffer,
    bindings: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    plasticity: wgpu::ComputePipeline,
    fusion: Option<wgpu::ComputePipeline>,
    fuse_projection: bool,
    commit: wgpu::ComputePipeline,
    quiescence: Option<quiescence::Observer>,
    pub(super) bounds: Option<[f32; 4]>,
}

impl MotionGuard {
    fn required_bytes(capacity: u32, limits: &wgpu::Limits) -> Result<u64> {
        let size = u64::from(capacity.max(1)) * 128;
        if size > limits.max_buffer_size || size > u64::from(limits.max_storage_buffer_binding_size)
        {
            return Err(Error::Solver(
                "swept proposals exceed device storage limits".into(),
            ));
        }
        Ok(16 + size)
    }

    fn new(sim: &MpmSimulation, max_bytes: u64) -> Result<Self> {
        let device = &sim.device;
        let bytes = Self::required_bytes(sim.cfg.max_particles, &device.limits())?;
        if bytes > max_bytes {
            return Err(Error::Solver(
                "swept proposals exceed the configured byte budget".into(),
            ));
        }
        let proposals = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("soil swept gather proposals"),
                size: bytes - 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }
        );
        let params = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("soil swept displacement bounds"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let entries = [0, 1, 2].map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if binding == 2 {
                    wgpu::BufferBindingType::Storage { read_only: false }
                } else {
                    wgpu::BufferBindingType::Uniform
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("soil swept gather uniforms"),
            entries: &entries,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("soil swept gather uniforms"),
            layout: &uniforms,
            entries: &[
                buffer_entry(0, &sim.uniform_buf),
                buffer_entry(1, &params),
                buffer_entry(2, &proposals),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("soil swept gather"),
            bind_group_layouts: &[
                Some(&uniforms),
                Some(&sim.layout_particles),
                Some(&sim.layout_grid),
                Some(&sim.layout_rigid),
            ],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("soil swept gather"),
            source: wgpu::ShaderSource::Wgsl(source().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("soil swept gather"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("g2p"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("SPARSE_GRID", f64::from(sim.sparse_grid))],
                ..Default::default()
            },
            cache: None,
        });
        let commit = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("soil swept gather publication"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("soil_motion_commit"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("SPARSE_GRID", f64::from(sim.sparse_grid))],
                ..Default::default()
            },
            cache: None,
        });
        let plasticity = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("soil swept proposal plasticity"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("soil_motion_plasticity"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("SPARSE_GRID", f64::from(sim.sparse_grid))],
                ..Default::default()
            },
            cache: None,
        });
        Ok(Self {
            params,
            proposals,
            bindings,
            pipeline,
            plasticity,
            fusion: None,
            fuse_projection: false,
            commit,
            quiescence: None,
            bounds: None,
        })
    }

    pub(super) fn upload(&self, queue: &wgpu::Queue) {
        if let Some(bounds) = self.bounds {
            molla_core::gpu_write_buffer!(queue, &self.params, 0, bytemuck::cast_slice(&bounds));
        }
    }

    pub(super) fn parameter_copy(&self) -> Option<(&wgpu::Buffer, &[f32; 4])> {
        self.bounds.as_ref().map(|bounds| (&self.params, bounds))
    }
}

impl MpmSimulation {
    pub(crate) fn soil_motion_guard_info(&self) -> (Option<[f32; 3]>, u64) {
        (
            self.motion_guard
                .as_ref()
                .and_then(|guard| guard.bounds.map(|b| [b[0], b[1], b[2]])),
            self.soil_motion_guard_bytes(),
        )
    }

    pub(crate) fn soil_motion_bound_enabled(&self) -> bool {
        self.motion_guard
            .as_ref()
            .is_some_and(|guard| guard.bounds.is_some())
    }

    pub(crate) fn configure_coupled_motion_bound(
        &mut self,
        bound: Option<[f64; 3]>,
        max_bytes: u64,
    ) -> Result<u64> {
        if bound.is_some_and(|b| b.iter().any(|v| !v.is_finite() || *v < 0.0)) {
            return Err(Error::Solver(
                "motion bounds must be finite and nonnegative".into(),
            ));
        }
        if self
            .motion_guard
            .as_ref()
            .is_some_and(|g| g.quiescence.is_some())
        {
            return Err(Error::Solver(
                "coupled motion bounds reject quiescence ownership".into(),
            ));
        }
        if self.soil_motion_guard_bytes() > max_bytes {
            return Err(Error::Solver(
                "retained swept proposals exceed the configured byte budget".into(),
            ));
        }
        if bound.is_some() {
            self.prepare_soil_motion_guard_with_budget(max_bytes)?;
        }
        self.set_soil_motion_bound(bound);
        Ok(self.soil_motion_guard_bytes())
    }

    pub(super) fn prepare_soil_motion_guard(&mut self) -> Result<()> {
        self.prepare_soil_motion_guard_with_budget(u64::MAX)
    }

    pub(super) fn prepare_soil_motion_guard_with_budget(&mut self, max_bytes: u64) -> Result<()> {
        #[cfg(test)]
        if self.use_fused_gather || self.bounded_reference.is_some() {
            return Err(Error::Solver(
                "swept motion guard requires standard gather coordinates".into(),
            ));
        }
        if self.soil_motion_guard_bytes() > max_bytes {
            return Err(Error::Solver(
                "retained swept proposals exceed the configured byte budget".into(),
            ));
        }
        if self.motion_guard.is_none() {
            self.motion_guard = Some(MotionGuard::new(self, max_bytes)?);
        }
        Ok(())
    }

    pub(super) fn soil_motion_guard_bytes(&self) -> u64 {
        self.motion_guard
            .as_ref()
            .map_or(0, |guard| guard.params.size() + guard.proposals.size())
    }

    pub(super) fn set_soil_motion_bound(&mut self, bound: Option<[f64; 3]>) {
        let bounds = bound.map(lower_bounds);
        if let Some(guard) = &mut self.motion_guard {
            if guard.bounds != bounds {
                guard.bounds = bounds;
                self.invalidate_prepared_steps();
            }
        } else {
            assert!(
                bound.is_none(),
                "swept gather must be prepared before committing"
            );
        }
    }

    pub(super) fn encode_gather(&self, pass: &mut wgpu::ComputePass<'_>, groups: u32) {
        self.encode_gather_dispatch(pass, groups, None);
    }

    pub(super) fn encode_projected_particles(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        indirect: &wgpu::Buffer,
        plastic: bool,
    ) {
        if plastic {
            self.encode_plasticity_dispatch(pass, 0, Some(indirect));
        } else {
            self.encode_gather_dispatch(pass, 0, Some(indirect));
        }
    }

    fn encode_gather_dispatch(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        groups: u32,
        indirect: Option<&wgpu::Buffer>,
    ) {
        if let Some(guard) = &self.motion_guard {
            if guard.bounds.is_some() {
                pass.set_bind_group(0, &guard.bindings, &[]);
                pass.set_pipeline(if guard.fuse_projection {
                    guard.fusion.as_ref().expect("prepared proposal fusion")
                } else {
                    &guard.pipeline
                });
                dispatch_particles(pass, groups, indirect);
                pass.set_bind_group(0, &self.bg_uniforms, &[]);
                return;
            }
        }
        pass.set_pipeline(&self.pipe_g2p);
        dispatch_particles(pass, groups, indirect);
    }

    pub(super) fn encode_plasticity(&self, pass: &mut wgpu::ComputePass<'_>, groups: u32) {
        self.encode_plasticity_dispatch(pass, groups, None);
    }

    fn encode_plasticity_dispatch(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        groups: u32,
        indirect: Option<&wgpu::Buffer>,
    ) {
        if let Some(guard) = &self.motion_guard {
            if guard.bounds.is_some() {
                pass.set_bind_group(0, &guard.bindings, &[]);
                if !guard.fuse_projection {
                    if let Some(cache) = guard
                        .quiescence
                        .as_ref()
                        .and_then(|o| o.projection_cache.as_ref())
                    {
                        pass.set_bind_group(0, &cache.bindings, &[]);
                        pass.set_pipeline(&cache.pipeline);
                    } else {
                        pass.set_bind_group(0, &guard.bindings, &[]);
                        pass.set_pipeline(&guard.plasticity);
                    }
                    dispatch_particles(pass, groups, indirect);
                }
                if let Some(observer) = &guard.quiescence {
                    pass.set_bind_group(0, &observer.bindings, &[]);
                    pass.set_pipeline(&observer.pipeline);
                } else {
                    pass.set_pipeline(&guard.commit);
                }
                dispatch_particles(pass, groups, indirect);
                pass.set_bind_group(0, &self.bg_uniforms, &[]);
                return;
            }
        }
        pass.set_pipeline(&self.pipe_plastic);
        dispatch_particles(pass, groups, indirect);
    }
}

fn dispatch_particles(
    pass: &mut wgpu::ComputePass<'_>,
    groups: u32,
    indirect: Option<&wgpu::Buffer>,
) {
    if let Some(buffer) = indirect {
        pass.dispatch_workgroups_indirect(buffer, 0);
    } else {
        pass.dispatch_workgroups(groups, 1, 1);
    }
    molla_core::gpu_metrics::record_dispatch();
}

#[cfg(test)]
#[path = "mpm_motion_guard_tests.rs"]
mod tests;
