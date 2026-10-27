//! Exact-input memoization of guarded constitutive projection.

use super::*;

pub(in crate::mpm) struct ProjectionCache {
    pub(in crate::mpm) records: wgpu::Buffer,
    pub(in crate::mpm) bindings: wgpu::BindGroup,
    pub(in crate::mpm) pipeline: wgpu::ComputePipeline,
}

fn source() -> String {
    let original = super::super::source();
    let entry = "@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)\nfn soil_motion_plasticity(@builtin(global_invocation_id) gid: vec3<u32>)";
    assert_eq!(original.matches(entry).count(), 1);
    let mut source = original.replace(entry, "fn soil_project_uncached(gid: vec3<u32>)");
    source.push_str(include_str!("../shaders/mpm_soil_projection_cache.wgsl"));
    source
}

fn bytes(capacity: u32, limits: &wgpu::Limits, budget: u64) -> Result<u64> {
    let bytes = u64::from(capacity.max(1)) * 112;
    if bytes > budget
        || bytes > limits.max_buffer_size
        || bytes > u64::from(limits.max_storage_buffer_binding_size)
    {
        return Err(Error::Solver(
            "soil projection cache exceeds budget/device limits".into(),
        ));
    }
    Ok(bytes)
}

impl ProjectionCache {
    pub(super) fn new(sim: &MpmSimulation, budget: u64) -> Result<Self> {
        let guard = sim.motion_guard.as_ref().unwrap();
        let device = &sim.device;
        let size = bytes(sim.cfg.max_particles, &device.limits(), budget)?;
        let records = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("soil projection cache"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("soil projection cache"),
            entries: &[0, 1, 2, 3, 4].map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding < 2 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding == 4,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }),
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("soil projection cache"),
            layout: &uniforms,
            entries: &[
                buffer_entry(0, &sim.uniform_buf),
                buffer_entry(1, &guard.params),
                buffer_entry(2, &guard.proposals),
                buffer_entry(3, &records),
                buffer_entry(4, &guard.quiescence.as_ref().unwrap().records),
            ],
        });
        let pipeline = Self::pipeline(sim, &uniforms, source());
        Ok(Self {
            records,
            bindings,
            pipeline,
        })
    }

    fn pipeline(
        sim: &MpmSimulation,
        uniforms: &wgpu::BindGroupLayout,
        source: String,
    ) -> wgpu::ComputePipeline {
        let device = &sim.device;
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("soil projection cache"),
            bind_group_layouts: &[
                Some(uniforms),
                Some(&sim.layout_particles),
                Some(&sim.layout_grid),
                Some(&sim.layout_rigid),
            ],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("soil projection cache"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("soil projection cache"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("soil_project_cached"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("SPARSE_GRID", f64::from(sim.sparse_grid))],
                ..Default::default()
            },
            cache: None,
        })
    }
}

impl MpmSimulation {
    pub(in crate::mpm) fn prepare_soil_projection_cache(&mut self, budget: u64) -> Result<()> {
        if self.soil_motion_projection_fused() {
            return Err(Error::Solver(
                "projection cache rejects fused proposals".into(),
            ));
        }
        let observer = self
            .motion_guard
            .as_ref()
            .and_then(|g| g.quiescence.as_ref())
            .ok_or_else(|| {
                Error::Solver("prepare swept quiescence before projection cache".into())
            })?;
        bytes(self.cfg.max_particles, &self.device.limits(), budget)?;
        if observer.projection_cache.is_some() {
            return Ok(());
        }
        let cache = ProjectionCache::new(self, budget)?;
        self.motion_guard
            .as_mut()
            .unwrap()
            .quiescence
            .as_mut()
            .unwrap()
            .projection_cache = Some(cache);
        self.invalidate_prepared_steps();
        Ok(())
    }

    pub(in crate::mpm) fn soil_projection_cache_bytes(&self) -> u64 {
        self.motion_guard
            .as_ref()
            .and_then(|g| g.quiescence.as_ref())
            .and_then(|o| o.projection_cache.as_ref())
            .map_or(0, |c| c.records.size())
    }

    pub(in crate::mpm) fn disable_soil_projection_cache(&mut self) {
        if let Some(observer) = self
            .motion_guard
            .as_mut()
            .and_then(|g| g.quiescence.as_mut())
        {
            observer.projection_cache = None;
        }
        self.invalidate_prepared_steps();
    }
}

#[cfg(test)]
#[path = "mpm_soil_projection_cache_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "mpm_soil_projection_cache_profile_tests.rs"]
mod profile_tests;
