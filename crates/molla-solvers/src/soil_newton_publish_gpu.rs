//! Independent publication checks for paired Newton impulse candidates.

use molla_compute::Engine;
use molla_core::{Error, Result};

#[path = "soil_admm_publication_probe.rs"]
mod active_probe;

pub struct SoilNewtonPublishInputs<'a> {
    /// Validated symmetric positive-definite generalized mass matrix.
    pub mass: &'a wgpu::Buffer,
    pub free_rates: &'a wgpu::Buffer,
    pub point_rows: &'a wgpu::Buffer,
    pub point_response: &'a wgpu::Buffer,
    pub contacts: &'a wgpu::Buffer,
    pub state: &'a wgpu::Buffer,
    pub upstream_status: &'a wgpu::Buffer,
    /// Newton status: only success, iteration limit and line-search rejection
    /// permit independent certification of the retained candidate.
    pub solver_status: &'a wgpu::Buffer,
}

/// Certifies the represented output impulses, velocities and rates at 1e-6.
/// Projects tolerance-sized outward wall velocities before certification and
/// recomputes their nonadhesive wall impulses from represented contact impulses.
/// Failed certification clears all outputs and rates. Mass/response preparation
/// must already be validated; this does not replace its SPD/response checks.
pub struct ExperimentalSoilNewtonPublishGpu {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    outputs: wgpu::Buffer,
    rates: wgpu::Buffer,
    status: wgpu::Buffer,
    bytes: u64,
    #[cfg(test)]
    parameters: wgpu::Buffer,
    #[cfg(test)]
    active_probe: bool,
}

fn invalid() -> Error {
    Error::Solver("soil Newton publication: invalid capacity, usage or device limit".into())
}

pub(crate) fn active_index_bytes(contacts: u32) -> u64 {
    4 * (u64::from(contacts.max(1)) + 1)
}

#[cfg(test)]
pub(crate) fn check_sorted_sum_gpu(
    engine: &Engine,
    root: &std::path::Path,
    read: impl Fn(&wgpu::Buffer) -> Vec<u8>,
) -> serde_json::Value {
    active_probe::check_sorted_sum_gpu(engine, root, read)
}

pub(crate) fn shader_source() -> String {
    format!(
        "{}\n{}\n{}\n{}",
        crate::soil_rate_encoding::arithmetic(64),
        include_str!("../shaders/soil_normal_rate_contract.wgsl"),
        include_str!("../shaders/soil_newton_publish.wgsl"),
        include_str!("../shaders/soil_newton_publish_machine.wgsl")
    )
}

pub(crate) fn resident_source() -> String {
    resident_source_with(include_str!("../shaders/soil_newton_publish_machine.wgsl"))
}

#[cfg(test)]
pub(crate) fn resident_serial_source() -> String {
    resident_source_with(include_str!(
        "../shaders/soil_newton_publish_machine_serial.wgsl"
    ))
}

fn resident_source_with(machine: &str) -> String {
    let source = format!(
        "{}\n{machine}",
        include_str!("../shaders/soil_newton_publish.wgsl")
    )
    .lines()
    .filter(|line| {
        ![
            "struct Contact ",
            "struct Six ",
            "struct State ",
            "struct Params ",
            "@group(",
            "@compute ",
            "fn finite(",
            "fn finite4(",
            "fn value(",
            "fn valid_six(",
        ]
        .iter()
        .any(|prefix| line.starts_with(prefix))
    })
    .collect::<Vec<_>>()
    .join("\n");
    let source = source.replace("@builtin(local_invocation_index) ", "");
    let mut result = String::new();
    let mut token = String::new();
    let emit = |token: &mut String, result: &mut String| {
        let replacement = match token.as_str() {
            "Output" => "PublicationOutput",
            "Status" => "PublicationStatus",
            "status" => "publication_status",
            "outputs" => "publication_outputs",
            "rates" => "publication_rates",
            "q" => "publication_q",
            "energy_parts" => "publication_energy_parts",
            "scales" => "publication_scales",
            "stop" => "publication_stop",
            "product" => "publication_product",
            "fault" => "publication_fault",
            "certify" => "publication_certify",
            other => other,
        };
        result.push_str(replacement);
        token.clear();
    };
    for ch in source.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            token.push(ch);
        } else {
            emit(&mut token, &mut result);
            result.push(ch);
        }
    }
    emit(&mut token, &mut result);
    assert_eq!(result.matches("solver_status[0]").count(), 2);
    result = result.replace("solver_status[0]", "atomicLoad(&report.fault)");
    format!(
        "{result}\n@group(0) @binding(17) var<storage,read> mass:array<f32>;\n@group(0) @binding(18) var<storage,read_write> publication_outputs:array<PublicationOutput>;\n@group(0) @binding(19) var<storage,read_write> publication_rates:array<vec2<f32>>;\n@group(0) @binding(20) var<storage,read_write> publication_status:PublicationStatus;\nfn certify_candidate(id:u32)->bool {{if id==0u {{atomicStore(&publication_status.publication_fault,0u);}}storageBarrier();publication_certify(id);return workgroupUniformLoad(&publication_stop)==0u;}}"
    )
}

impl ExperimentalSoilNewtonPublishGpu {
    pub fn required_bytes(dofs: u32, contacts: u32) -> Result<u64> {
        if !(1..=32).contains(&dofs) {
            return Err(invalid());
        }
        Ok(80 * u64::from(contacts.max(1)) + 8 * u64::from(dofs) + 48)
    }
    pub fn new(
        engine: &Engine,
        input: SoilNewtonPublishInputs<'_>,
        dofs: u32,
        contacts: u32,
        max_owned_bytes: u64,
    ) -> Result<Self> {
        Self::new_with_program(engine, input, dofs, contacts, max_owned_bytes, |source| {
            source
        })
    }

    pub(crate) fn new_with_program(
        engine: &Engine,
        input: SoilNewtonPublishInputs<'_>,
        dofs: u32,
        contacts: u32,
        max_owned_bytes: u64,
        program: impl FnOnce(String) -> String,
    ) -> Result<Self> {
        Self::new_internal(
            engine,
            input,
            dofs,
            contacts,
            max_owned_bytes,
            program,
            None,
            [0, 0],
        )
    }

    #[cfg(test)]
    pub(crate) fn new_active_probe(
        engine: &Engine,
        input: SoilNewtonPublishInputs<'_>,
        dofs: u32,
        contacts: u32,
        max_owned_bytes: u64,
        indices: &wgpu::Buffer,
    ) -> Result<Self> {
        Self::new_internal(
            engine,
            input,
            dofs,
            contacts,
            max_owned_bytes,
            active_probe::source,
            Some(indices),
            [0, 0],
        )
    }

    pub(crate) fn new_guarded_active(
        engine: &Engine,
        input: SoilNewtonPublishInputs<'_>,
        dofs: u32,
        contacts: u32,
        max_owned_bytes: u64,
        indices: &wgpu::Buffer,
    ) -> Result<Self> {
        Self::new_internal(
            engine,
            input,
            dofs,
            contacts,
            max_owned_bytes,
            active_probe::sorted_source,
            Some(indices),
            [1, 1],
        )
    }

    #[cfg(test)]
    pub(crate) fn set_active_probe(&self, engine: &Engine, enabled: bool) {
        assert!(self.active_probe);
        engine.queue().unwrap().write_buffer(
            &self.parameters,
            8,
            &u32::from(enabled).to_le_bytes(),
        );
    }

    #[cfg(test)]
    pub(crate) fn set_numerical_probe(&self, engine: &Engine, enabled: bool) {
        assert!(self.active_probe);
        engine.queue().unwrap().write_buffer(
            &self.parameters,
            12,
            &u32::from(enabled).to_le_bytes(),
        );
    }

    fn new_internal(
        engine: &Engine,
        input: SoilNewtonPublishInputs<'_>,
        dofs: u32,
        contacts: u32,
        max_owned_bytes: u64,
        program: impl FnOnce(String) -> String,
        indices: Option<&wgpu::Buffer>,
        flags: [u32; 2],
    ) -> Result<Self> {
        engine.check_device()?;
        let device = engine.device().ok_or_else(invalid)?;
        let limits = device.limits();
        let bytes = Self::required_bytes(dofs, contacts)?;
        let n = u64::from(dofs);
        let c = u64::from(contacts.max(1));
        let borrowed = [
            input.mass,
            input.free_rates,
            input.point_rows,
            input.point_response,
            input.contacts,
            input.state,
            input.upstream_status,
            input.solver_status,
        ];
        let sizes = [80 * c, 8 * n, 32, 16];
        if bytes > max_owned_bytes
            || limits.max_storage_buffers_per_shader_stage < 11 + u32::from(indices.is_some())
            || limits.max_bindings_per_bind_group < 12 + u32::from(indices.is_some())
            || limits.max_uniform_buffers_per_shader_stage < 1
            || limits.max_compute_invocations_per_workgroup < 64
            || limits.max_compute_workgroup_size_x < 64
            || limits.max_compute_workgroup_storage_size < 2048
            || borrowed
                .iter()
                .zip([
                    4 * n * n,
                    8 * n,
                    16 * c * n,
                    16 * c * n,
                    64 * c,
                    64 * c,
                    4,
                    4,
                ])
                .any(|(b, s)| {
                    b.size() < s
                        || b.size() > limits.max_storage_buffer_binding_size
                        || !b.usage().contains(wgpu::BufferUsages::STORAGE)
                })
            || sizes
                .iter()
                .any(|&s| s > limits.max_buffer_size || s > limits.max_storage_buffer_binding_size)
            || indices.is_some_and(|b| {
                b.size() < active_index_bytes(contacts)
                    || b.size() > limits.max_storage_buffer_binding_size
                    || !b.usage().contains(wgpu::BufferUsages::STORAGE)
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
                        label: Some("Newton publication storage"),
                        size,
                        usage: if i == 3 {
                            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST
                        } else {
                            wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_SRC
                                | wgpu::BufferUsages::COPY_DST
                        },
                        mapped_at_creation: false
                    }
                )
            })
            .collect();
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &buffers[3],
            0,
            bytemuck::cast_slice(&[dofs, contacts, flags[0], flags[1]])
        );
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Newton publication layout"),
            entries: &(0..12 + u32::from(indices.is_some()))
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 11 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: binding < 8 || binding == 12,
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
            label: Some("Newton publication pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Newton publication"),
            source: wgpu::ShaderSource::Wgsl(program(shader_source()).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Newton publication"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("certify"),
            compilation_options: Default::default(),
            cache: None,
        });
        let mut bindings = vec![
            input.mass,
            input.free_rates,
            input.point_rows,
            input.point_response,
            input.contacts,
            input.state,
            input.upstream_status,
            input.solver_status,
            &buffers[0],
            &buffers[1],
            &buffers[2],
            &buffers[3],
        ];
        if let Some(indices) = indices {
            bindings.push(indices);
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Newton publication bindings"),
            layout: &layout,
            entries: &bindings
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        Ok(Self {
            pipeline,
            group,
            outputs: buffers[0].clone(),
            rates: buffers[1].clone(),
            status: buffers[2].clone(),
            bytes,
            #[cfg(test)]
            parameters: buffers[3].clone(),
            #[cfg(test)]
            active_probe: indices.is_some(),
        })
    }
    pub fn encode_reset(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.status, 0, None);
    }
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }
    pub fn outputs_buffer(&self) -> &wgpu::Buffer {
        &self.outputs
    }
    pub fn rates_buffer(&self) -> &wgpu::Buffer {
        &self.rates
    }
    /// Sticky fault, first failing contact (one-based), active count, solver
    /// fault, energy change, energy scale and two padding words. Fault bits:
    /// input=1, nonfinite=2, momentum=4, contact=8, wall=16, energy=32.
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        &self.status
    }
    pub fn owned_buffer_bytes(&self) -> u64 {
        self.bytes
    }
}
