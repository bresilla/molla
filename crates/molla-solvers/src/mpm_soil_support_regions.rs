//! Bounded GPU retention seeds from conservative local region boxes.

use super::*;

pub(super) struct RegionSeeds {
    pub(super) boxes: wgpu::Buffer,
    pub(super) bindings: wgpu::BindGroup,
    pub(super) pipeline: wgpu::ComputePipeline,
    pub(super) capacity: u32,
}

fn size(capacity: u32, budget: u64, limits: &wgpu::Limits) -> Result<u64> {
    let bytes = u64::from(capacity) * 32;
    if !(1..=64).contains(&capacity)
        || bytes > budget
        || bytes > limits.max_buffer_size
        || bytes > u64::from(limits.max_storage_buffer_binding_size)
    {
        return Err(Error::Solver(
            "soil support region capacity/budget exceeded".into(),
        ));
    }
    Ok(bytes)
}

impl RegionSeeds {
    fn new(sim: &MpmSimulation, query: &SupportQuery, capacity: u32, budget: u64) -> Result<Self> {
        let device = &sim.device;
        let bytes = size(capacity, budget, &device.limits())?;
        let boxes = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("soil support region boxes"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("soil support region seeds"),
            entries: &[0, 1, 4, 5].map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding != 4,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            }),
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("soil support region seeds"),
            layout: &layout,
            entries: &[
                buffer_entry(0, &query.params),
                buffer_entry(1, &sim.p_pos),
                buffer_entry(4, &query.result),
                buffer_entry(5, &boxes),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("soil support region seeds"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/mpm_soil_support_regions.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("soil support region seeds"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("region_seeds"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            boxes,
            bindings,
            pipeline,
            capacity,
        })
    }
}

impl MpmSimulation {
    pub(in crate::mpm) fn prepare_soil_support_regions(
        &mut self,
        capacity: u32,
        budget: u64,
    ) -> Result<()> {
        size(capacity, budget, &self.device.limits())?;
        let query = self
            .motion_guard
            .as_ref()
            .and_then(|g| g.quiescence.as_ref())
            .and_then(|o| o.support.as_ref())
            .ok_or_else(|| Error::Solver("prepare soil support before region seeds".into()))?;
        if query
            .regions
            .as_ref()
            .is_some_and(|r| r.capacity == capacity)
        {
            return Ok(());
        }
        let regions = RegionSeeds::new(self, query, capacity, budget)?;
        self.motion_guard
            .as_mut()
            .unwrap()
            .quiescence
            .as_mut()
            .unwrap()
            .support
            .as_mut()
            .unwrap()
            .regions = Some(regions);
        Ok(())
    }
    pub(in crate::mpm) fn soil_support_region_capacity(&self) -> u32 {
        self.motion_guard
            .as_ref()
            .and_then(|g| g.quiescence.as_ref())
            .and_then(|o| o.support.as_ref())
            .and_then(|q| q.regions.as_ref())
            .map_or(0, |r| r.capacity)
    }
    pub(in crate::mpm) fn soil_support_region_bytes(&self) -> u64 {
        u64::from(self.soil_support_region_capacity()) * 32
    }
    pub(in crate::mpm) fn read_soil_region_support(
        &mut self,
        engine: &Engine,
        snapshot: &MpmSoilQuiescenceSnapshot,
        minimum_quiet_steps: u32,
        max_particles: u32,
        rounds: u32,
        boxes: &[[f32; 8]],
    ) -> Result<MpmSoilRetainedSupportSnapshot> {
        self.validate_soil_quiescence(snapshot)?;
        if !(1..=64).contains(&rounds) || snapshot.identity().is_none() {
            return Err(Error::Solver(
                "soil region support requires identities and 1..=64 rounds".into(),
            ));
        }
        let words = self.read_soil_support_words(
            engine,
            snapshot,
            minimum_quiet_steps,
            max_particles,
            Some(rounds),
            None,
            Some(boxes),
        )?;
        retained::from_region_words(snapshot, minimum_quiet_steps, rounds, &words)
    }

    pub(in crate::mpm) fn try_request_soil_region_support(
        &mut self,
        engine: &Engine,
        snapshot: &MpmSoilQuiescenceSnapshot,
        minimum_quiet_steps: u32,
        max_particles: u32,
        rounds: u32,
        boxes: &[[f32; 8]],
    ) -> Result<bool> {
        self.validate_soil_quiescence(snapshot)?;
        if !(1..=64).contains(&rounds) || snapshot.identity().is_none() {
            return Err(Error::Solver(
                "soil region support requires identities and 1..=64 rounds".into(),
            ));
        }
        self.request_soil_support_words(
            engine,
            snapshot,
            minimum_quiet_steps,
            max_particles,
            Some(rounds),
            None,
            Some(boxes),
            Some(SupportRequest::Region(rounds)),
        )
    }

    pub(in crate::mpm) fn poll_soil_region_support(
        &mut self,
        engine: &Engine,
    ) -> Result<Option<MpmSoilRetainedSupportSnapshot>> {
        self.poll_soil_support_words(engine, false, Some(SupportKind::Region))?
            .map(|(pending, words)| {
                let Some(SupportRequest::Region(rounds)) = pending.request else {
                    unreachable!("checked support request kind");
                };
                retained::from_region_words(
                    &pending.snapshot.expect("asynchronous support provenance"),
                    pending.minimum_quiet_steps,
                    rounds,
                    &words,
                )
            })
            .transpose()
    }
}

#[cfg(test)]
#[path = "mpm_soil_support_regions_tests.rs"]
mod tests;
