//! Experimental resident mass-metric ADMM with independent guarded publication.

use crate::soil_newton_publish_gpu::{
    ExperimentalSoilNewtonPublishGpu, SoilNewtonPublishInputs, active_index_bytes,
};
use molla_compute::Engine;
use molla_core::{Error, Result};

#[cfg(test)]
#[path = "soil_admm_shared_cache.rs"]
mod shared_cache;

#[cfg(test)]
#[path = "soil_admm_shared_iterate.rs"]
mod shared_iterate;

#[path = "soil_admm_response_cache.rs"]
mod response_cache;

#[derive(Clone, Copy, Debug)]
pub struct SoilAdmmCapacity {
    pub dofs: u32,
    pub contacts: u32,
    pub iterations: u32,
    pub max_owned_bytes: u64,
}

/// Prepared mass/response inputs must satisfy the upstream SPD/response contract.
#[derive(Clone, Copy)]
pub struct SoilAdmmInputs<'a> {
    pub mass: &'a wgpu::Buffer,
    pub free_rates: &'a wgpu::Buffer,
    pub point_rows: &'a wgpu::Buffer,
    pub point_response: &'a wgpu::Buffer,
    pub contacts: &'a wgpu::Buffer,
    pub upstream_status: &'a wgpu::Buffer,
}

/// Cold-starts each solve and publishes only physically and numerically certified output.
/// Publication faults remain latched until `encode_reset`; scratch stays device-resident.
pub struct ExperimentalSoilAdmmGpu {
    program: std::sync::Arc<AdmmProgram>,
    group: wgpu::BindGroup,
    _buffers: Vec<wgpu::Buffer>,
    publication: ExperimentalSoilNewtonPublishGpu,
    bytes: u64,
}

fn invalid() -> Error {
    Error::Solver("soil ADMM: invalid capacity, buffer usage, budget or device limit".into())
}

struct AdmmProgram {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

fn program_key(source: &str) -> String {
    format!(
        "soil-admm-v1;layout=10;wgsl={}",
        blake3::hash(source.as_bytes())
    )
}

fn shader_source(dofs: u32) -> Result<String> {
    if !(1..=32).contains(&dofs) {
        return Err(invalid());
    }
    Ok(include_str!("../shaders/soil_admm_probe.wgsl").replace(
        "staged_responses:array<vec4<f32>,2048>",
        &format!("staged_responses:array<vec4<f32>,{}>", 64 * dofs.max(4)),
    ))
}

fn create_program(engine: &Engine, source: &str) -> Result<AdmmProgram> {
    let device = engine.device().ok_or_else(invalid)?;
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("resident ADMM layout"),
        entries: &(0..10)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 7 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding < 4,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect::<Vec<_>>(),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("resident ADMM pipeline layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("resident mass-metric ADMM"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("resident ADMM"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("admm"),
        compilation_options: Default::default(),
        cache: engine.pipeline_cache(),
    });
    Ok(AdmmProgram { layout, pipeline })
}

impl ExperimentalSoilAdmmGpu {
    pub fn required_bytes(capacity: SoilAdmmCapacity) -> Result<u64> {
        if !(1..=32).contains(&capacity.dofs)
            || capacity.contacts > 16384
            || !(1..=64).contains(&capacity.iterations)
        {
            return Err(invalid());
        }
        Ok(304 * u64::from(capacity.contacts.max(1))
            + active_index_bytes(capacity.contacts)
            + 8 * u64::from(capacity.dofs)
            + 96)
    }

    pub fn new(
        engine: &Engine,
        input: SoilAdmmInputs<'_>,
        capacity: SoilAdmmCapacity,
    ) -> Result<Self> {
        Self::new_with_source(
            engine,
            input,
            capacity,
            &response_cache::source(capacity.dofs)?,
        )
    }

    fn new_with_source(
        engine: &Engine,
        input: SoilAdmmInputs<'_>,
        capacity: SoilAdmmCapacity,
        source: &str,
    ) -> Result<Self> {
        let bytes = Self::required_bytes(capacity)?;
        engine.check_device()?;
        let device = engine.device().ok_or_else(invalid)?;
        let limits = device.limits();
        let n = u64::from(capacity.dofs);
        let c = u64::from(capacity.contacts.max(1));
        let sizes = [
            96 * c,
            64 * c,
            32,
            16,
            active_index_bytes(capacity.contacts),
            64 * c,
        ];
        let borrowed = [
            input.mass,
            input.free_rates,
            input.point_rows,
            input.point_response,
            input.contacts,
            input.upstream_status,
        ];
        let borrowed_sizes = [4 * n * n, 8 * n, 16 * c * n, 16 * c * n, 64 * c, 4];
        if bytes > capacity.max_owned_bytes
            || limits.max_storage_buffers_per_shader_stage < 12
            || limits.max_bindings_per_bind_group < 13
            || limits.max_uniform_buffers_per_shader_stage < 1
            || limits.max_compute_invocations_per_workgroup < 64
            || limits.max_compute_workgroup_size_x < 64
            || limits.max_compute_workgroup_storage_size < 38400
            || sizes.into_iter().chain([80 * c, 8 * n]).any(|size| {
                size > limits.max_buffer_size || size > limits.max_storage_buffer_binding_size
            })
            || borrowed.iter().zip(borrowed_sizes).any(|(buffer, size)| {
                buffer.size() < size
                    || buffer.size() > limits.max_storage_buffer_binding_size
                    || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            })
        {
            return Err(invalid());
        }
        let buffers: Vec<_> = sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                molla_core::gpu_create_buffer!(
                    device,
                    &wgpu::BufferDescriptor {
                        label: Some("resident ADMM storage"),
                        size,
                        usage: if i == 3 {
                            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
                        } else {
                            wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_SRC
                                | wgpu::BufferUsages::COPY_DST
                        },
                        mapped_at_creation: false,
                    }
                )
            })
            .collect();
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &buffers[3],
            0,
            bytemuck::cast_slice(&[capacity.dofs, capacity.contacts, capacity.iterations, 3u32])
        );
        let key = program_key(source);
        #[cfg(not(target_arch = "wasm32"))]
        let program = engine.cached_gpu_program(&key, |_| create_program(engine, source))?;
        #[cfg(target_arch = "wasm32")]
        let program = std::sync::Arc::new(create_program(engine, source)?);
        let bindings = [
            input.free_rates,
            input.point_rows,
            input.point_response,
            input.contacts,
            &buffers[0],
            &buffers[1],
            &buffers[2],
            &buffers[3],
            &buffers[4],
            &buffers[5],
        ];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("resident ADMM bindings"),
            layout: &program.layout,
            entries: &bindings
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let publication = ExperimentalSoilNewtonPublishGpu::new_guarded_active(
            engine,
            SoilNewtonPublishInputs {
                mass: input.mass,
                free_rates: input.free_rates,
                point_rows: input.point_rows,
                point_response: input.point_response,
                contacts: input.contacts,
                state: &buffers[1],
                upstream_status: input.upstream_status,
                solver_status: &buffers[2],
            },
            capacity.dofs,
            capacity.contacts,
            ExperimentalSoilNewtonPublishGpu::required_bytes(capacity.dofs, capacity.contacts)?,
            &buffers[4],
        )?;
        debug_assert_eq!(
            bytes,
            sizes.iter().sum::<u64>() + publication.owned_buffer_bytes()
        );
        Ok(Self {
            program,
            group,
            _buffers: buffers,
            publication,
            bytes,
        })
    }

    pub fn encode_reset(&self, encoder: &mut wgpu::CommandEncoder) {
        self.publication.encode_reset(encoder);
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_solve(encoder);
        self.encode_publication(encoder);
    }

    pub(crate) fn encode_solve(&self, encoder: &mut wgpu::CommandEncoder) {
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.program.pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        molla_core::gpu_metrics::record_dispatch();
    }

    pub(crate) fn encode_publication(&self, encoder: &mut wgpu::CommandEncoder) {
        self.publication.encode(encoder);
    }

    pub fn outputs_buffer(&self) -> &wgpu::Buffer {
        self.publication.outputs_buffer()
    }
    pub fn rates_buffer(&self) -> &wgpu::Buffer {
        self.publication.rates_buffer()
    }
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        self.publication.status_buffer()
    }
    pub fn owned_buffer_bytes(&self) -> u64 {
        self.bytes
    }

    pub(crate) fn solver_status_buffer(&self) -> &wgpu::Buffer {
        &self._buffers[2]
    }

    #[cfg(test)]
    pub(crate) fn onchip_slots(dofs: u32) -> Result<u32> {
        shared_cache::slots(dofs)
    }

    #[cfg(test)]
    pub(crate) fn new_global_staging_control(
        engine: &Engine,
        input: SoilAdmmInputs<'_>,
        capacity: SoilAdmmCapacity,
    ) -> Result<Self> {
        Self::new_with_source(engine, input, capacity, &shader_source(capacity.dofs)?)
    }

    #[cfg(test)]
    pub(crate) fn response_cache_slots(dofs: u32) -> Result<u32> {
        response_cache::slots(dofs)
    }

    #[cfg(test)]
    pub(crate) fn new_response_cache_probe(
        engine: &Engine,
        input: SoilAdmmInputs<'_>,
        capacity: SoilAdmmCapacity,
    ) -> Result<Self> {
        Self::new_with_source(
            engine,
            input,
            capacity,
            &response_cache::source(capacity.dofs)?,
        )
    }

    #[cfg(test)]
    pub(crate) fn new_shared_iterate_probe(
        engine: &Engine,
        input: SoilAdmmInputs<'_>,
        capacity: SoilAdmmCapacity,
    ) -> Result<Self> {
        Self::new_with_source(
            engine,
            input,
            capacity,
            &shared_iterate::source(capacity.dofs)?,
        )
    }

    #[cfg(test)]
    pub(crate) fn new_padded_probe(
        engine: &Engine,
        input: SoilAdmmInputs<'_>,
        capacity: SoilAdmmCapacity,
    ) -> Result<Self> {
        Self::new_with_source(
            engine,
            input,
            capacity,
            &shared_cache::padded_source(capacity.dofs)?,
        )
    }

    #[cfg(test)]
    pub(crate) fn new_onchip_probe(
        engine: &Engine,
        input: SoilAdmmInputs<'_>,
        capacity: SoilAdmmCapacity,
    ) -> Result<Self> {
        Self::new_with_source(
            engine,
            input,
            capacity,
            &shared_cache::source(capacity.dofs)?,
        )
    }

    #[cfg(test)]
    pub(crate) fn new_full_staging_control(
        engine: &Engine,
        input: SoilAdmmInputs<'_>,
        capacity: SoilAdmmCapacity,
    ) -> Result<Self> {
        Self::new_with_source(
            engine,
            input,
            capacity,
            include_str!("../shaders/soil_admm_probe.wgsl"),
        )
    }

    #[cfg(test)]
    pub(crate) fn shares_program(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.program, &other.program)
    }

    #[cfg(test)]
    pub(crate) fn diagnostic_buffers(&self) -> [&wgpu::Buffer; 2] {
        [&self._buffers[1], &self._buffers[2]]
    }

    #[cfg(test)]
    pub(crate) fn active_indices_buffer(&self) -> &wgpu::Buffer {
        &self._buffers[4]
    }

    #[cfg(test)]
    pub(crate) fn scratch_buffers(&self) -> [&wgpu::Buffer; 5] {
        [0, 1, 2, 4, 5].map(|i| &self._buffers[i])
    }

    #[cfg(test)]
    pub(crate) fn encode_publication_probe(&self, encoder: &mut wgpu::CommandEncoder) {
        self.publication.encode(encoder);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admm_resident_staging_capacity_contract() {
        let original = include_str!("../shaders/soil_admm_probe.wgsl");
        let declaration = "staged_responses:array<vec4<f32>,2048>";
        assert_eq!(original.matches(declaration).count(), 1);
        for dofs in 1..=32 {
            let source = shader_source(dofs).unwrap();
            let replacement = format!("staged_responses:array<vec4<f32>,{}>", 64 * dofs.max(4));
            assert_eq!(source.replace(&replacement, declaration), original);
            let module = naga::front::wgsl::parse_str(&source).unwrap();
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            let (_, variable) = module
                .global_variables
                .iter()
                .find(|(_, v)| v.name.as_deref() == Some("staged_responses"))
                .unwrap();
            let naga::TypeInner::Array { size, stride, .. } = module.types[variable.ty].inner
            else {
                panic!("staged response array")
            };
            assert_eq!(
                size,
                naga::ArraySize::Constant((64 * dofs.max(4)).try_into().unwrap())
            );
            assert_eq!(stride, 16);
            assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
        }
        assert_eq!(shader_source(32).unwrap(), original);
        assert_ne!(
            program_key(&shader_source(10).unwrap()),
            program_key(original)
        );
        assert_ne!(
            program_key(&shader_source(10).unwrap()),
            program_key(&shader_source(9).unwrap())
        );
        for dofs in [0, 33, u32::MAX] {
            assert!(shader_source(dofs).is_err());
        }
        if let Some(path) = std::env::var_os("MOLLA_ADMM_STAGING_ARTIFACT") {
            std::fs::write(path, shader_source(10).unwrap()).unwrap();
        }
    }

    #[test]
    fn admm_resident_program_key_contract() {
        let source = include_str!("../shaders/soil_admm_probe.wgsl");
        assert_eq!(program_key(source), program_key(source));
        assert_ne!(program_key(source), program_key(&format!("{source}\n")));
        assert!(program_key(source).starts_with("soil-admm-v1;layout=10;wgsl="));
    }

    #[test]
    fn admm_resident_empty_binding_contract() {
        let module =
            naga::front::wgsl::parse_str(include_str!("../shaders/soil_admm_probe.wgsl")).unwrap();
        let active = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("Active"))
            .unwrap()
            .1;
        let naga::TypeInner::Struct { span, members } = &active.inner else {
            panic!("active index layout")
        };
        assert_eq!(*span, 8);
        assert_eq!(members[1].offset, 4);
        assert_eq!(active_index_bytes(0), u64::from(*span));
        assert_eq!(active_index_bytes(1), 8);
        assert_eq!(active_index_bytes(4096), 16388);
        assert_eq!(
            ExperimentalSoilAdmmGpu::required_bytes(SoilAdmmCapacity {
                dofs: 1,
                contacts: 0,
                iterations: 64,
                max_owned_bytes: u64::MAX
            })
            .unwrap(),
            416
        );
    }

    #[test]
    fn admm_resident_capacity_contract() {
        for n in [1u32, 10, 32] {
            for c in [0u32, 1, 4096, 16384] {
                let capacity = SoilAdmmCapacity {
                    dofs: n,
                    contacts: c,
                    iterations: 64,
                    max_owned_bytes: u64::MAX,
                };
                let local = 224 * u64::from(c.max(1)) + active_index_bytes(c) + 48;
                assert_eq!(
                    ExperimentalSoilAdmmGpu::required_bytes(capacity).unwrap(),
                    local + ExperimentalSoilNewtonPublishGpu::required_bytes(n, c).unwrap()
                );
            }
        }
        for (n, c, i) in [
            (0, 0, 64),
            (33, 0, 64),
            (1, 16385, 64),
            (1, 1, 0),
            (1, 1, 65),
            (u32::MAX, u32::MAX, u32::MAX),
        ] {
            assert!(
                ExperimentalSoilAdmmGpu::required_bytes(SoilAdmmCapacity {
                    dofs: n,
                    contacts: c,
                    iterations: i,
                    max_owned_bytes: u64::MAX
                })
                .is_err()
            );
        }
    }
}
