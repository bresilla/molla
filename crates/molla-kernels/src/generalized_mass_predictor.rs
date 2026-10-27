//! Experimental retained dense generalized acceleration solve.

use molla_core::{Error, Result};

/// Solves M a = drive - bias with separate f64 solves and high/low f32 output.
///
/// M is row-major f32 or explicitly paired, exactly symmetric and positive definite. Drive and bias
/// are high/low f32 pairs in the same generalized basis, with 1..=32 DOFs.
/// The caller orders current M/drive/bias writes before each dispatch. This is
/// an algebra component, not a free-joint bias producer or integrated predictor.
pub struct ExperimentalMassPredictor {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    output: wgpu::Buffer,
    status: wgpu::Buffer,
}

fn source(width: u32, upstream: bool) -> String {
    let text = include_str!("../shaders/generalized_mass_predictor.wgsl")
        .replace("__WIDTH__", &width.to_string());
    if !upstream {
        return text;
    }
    let guard = "    if status[0] != 0u { return; }";
    assert_eq!(text.matches(guard).count(), 1);
    format!(
        "@group(0) @binding(5) var<storage, read> upstream_status: array<u32>;\n{}",
        text.replace(
            guard,
            &format!("{guard}\n    if upstream_status[0] != 0u {{ status[0] = 4u; return; }}")
        )
    )
}

fn paired_source(text: String) -> String {
    text.replace("mass: array<f32>", "mass: array<vec2<f32>>")
        .replace("mass[i * N + j]", "pair_value(mass[i * N + j])")
        .replace("mass[j * N + i]", "pair_value(mass[j * N + i])")
}

fn sizes(width: u32, budget: u64) -> Result<[u64; 5]> {
    if !(1..=32).contains(&width) || budget < u64::from(width) * 8 + 4 {
        return Err(Error::Gpu(
            "invalid mass predictor width or output budget".into(),
        ));
    }
    let n = u64::from(width);
    Ok([4 * n * n, 8 * n, 8 * n, 8 * n, 4])
}

impl ExperimentalMassPredictor {
    /// Allocates output/status only; budget bounds those retained GPU buffer bytes.
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        mass: &wgpu::Buffer,
        drive: &wgpu::Buffer,
        bias: &wgpu::Buffer,
        output_budget: u64,
    ) -> Result<Self> {
        Self::build(device, width, mass, drive, bias, None, output_budget, false)
    }

    /// Latches status 4 and preserves output when the ordered bias producer has a fault.
    pub fn new_with_bias_status(
        device: &wgpu::Device,
        width: u32,
        mass: &wgpu::Buffer,
        drive: &wgpu::Buffer,
        bias: &wgpu::Buffer,
        bias_status: &wgpu::Buffer,
        output_budget: u64,
    ) -> Result<Self> {
        Self::build(
            device,
            width,
            mass,
            drive,
            bias,
            Some(bias_status),
            output_budget,
            false,
        )
    }

    /// Solves paired Gram entries with the same sticky upstream guard and output contract.
    pub fn new_with_paired_mass_and_bias_status(
        device: &wgpu::Device,
        width: u32,
        mass: &wgpu::Buffer,
        drive: &wgpu::Buffer,
        bias: &wgpu::Buffer,
        bias_status: &wgpu::Buffer,
        output_budget: u64,
    ) -> Result<Self> {
        Self::build(
            device,
            width,
            mass,
            drive,
            bias,
            Some(bias_status),
            output_budget,
            true,
        )
    }

    fn build(
        device: &wgpu::Device,
        width: u32,
        mass: &wgpu::Buffer,
        drive: &wgpu::Buffer,
        bias: &wgpu::Buffer,
        bias_status: Option<&wgpu::Buffer>,
        output_budget: u64,
        paired: bool,
    ) -> Result<Self> {
        let mut required = sizes(width, output_budget)?.to_vec();
        if paired {
            required[0] *= 2;
        }
        if bias_status.is_some() {
            required.push(4);
        }
        let invalid = || Error::Gpu("invalid mass predictor device or bindings".into());
        let limits = device.limits();
        let inputs = [mass, drive, bias];
        if !device.features().contains(wgpu::Features::SHADER_F64)
            || limits.max_storage_buffers_per_shader_stage < required.len() as u32
            || limits.max_bindings_per_bind_group < required.len() as u32
            || required.iter().any(|&size| {
                size > limits.max_storage_buffer_binding_size || size > limits.max_buffer_size
            })
            || inputs
                .iter()
                .zip(required.iter().copied())
                .any(|(buffer, size)| {
                    buffer.size() < size || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
                })
            || bias_status.is_some_and(|buffer| {
                buffer.size() < 4 || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
            })
        {
            return Err(invalid());
        }
        let entries: Vec<_> = required
            .iter()
            .enumerate()
            .map(|(i, &size)| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: i < 3 || i == 5,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(size),
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("experimental mass predictor"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("experimental mass predictor"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let text = source(width, bias_status.is_some());
        let text = if paired { paired_source(text) } else { text };
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("experimental mass predictor"),
            source: wgpu::ShaderSource::Wgsl(text.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("experimental mass predictor"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("predict"),
            compilation_options: Default::default(),
            cache: None,
        });
        let create = |size, reset| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("experimental mass predictor output"),
                    size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | if reset {
                            wgpu::BufferUsages::COPY_DST
                        } else {
                            wgpu::BufferUsages::empty()
                        },
                    mapped_at_creation: false,
                }
            )
        };
        let output = create(required[3], false);
        let status = create(4, true);
        let mut buffers = vec![mass, drive, bias, &output, &status];
        if let Some(upstream) = bias_status {
            buffers.push(upstream);
        }
        let entries: Vec<_> = buffers
            .iter()
            .zip(required)
            .enumerate()
            .map(|(i, (buffer, size))| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(size),
                }),
            })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("experimental mass predictor"),
            layout: &layout,
            entries: &entries,
        });
        Ok(Self {
            pipeline,
            group,
            output,
            status,
        })
    }

    /// Encodes one retained dispatch with no host transfers or buffer allocations.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }

    /// Clears sticky status without changing the last published acceleration.
    pub fn reset(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.status, 0, None);
    }

    /// Returns high/low f32 pairs, unchanged on a fault or a sticky-fault dispatch.
    pub fn acceleration(&self) -> &wgpu::Buffer {
        &self.output
    }

    /// Returns 0 no fault, 1 invalid input, 2 non-SPD, 3 failed solve, 4 upstream fault.
    pub fn status(&self) -> &wgpu::Buffer {
        &self.status
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mass_predictor_source_contract() {
        for width in [1, 10, 32] {
            for upstream in [false, true] {
                let text = source(width, upstream);
                for text in [text.clone(), paired_source(text)] {
                    let module = naga::front::wgsl::parse_str(&text).unwrap();
                    naga::valid::Validator::new(
                        naga::valid::ValidationFlags::all(),
                        naga::valid::Capabilities::all(),
                    )
                    .validate(&module)
                    .unwrap();
                }
            }
            assert_eq!(
                sizes(width, 8 * u64::from(width) + 4).unwrap()[0],
                4 * u64::from(width).pow(2)
            );
            assert!(sizes(width, 8 * u64::from(width) + 3).is_err());
        }
        assert!(sizes(0, u64::MAX).is_err());
        assert!(sizes(33, u64::MAX).is_err());
    }
}
