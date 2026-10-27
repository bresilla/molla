//! Retained generalized velocity prediction before coordinate drift.

use crate::{
    core::{Error, Result},
    sim::Model,
};
use molla_kernels::{ExperimentalBiasBindings, ExperimentalBiasKernel, ExperimentalMassPredictor};

/// Disjoint output buffers in the paired Featherstone generalized-rate layout.
pub struct GeneralizedPredictionOutputs<'a> {
    pub paired_rates: &'a wgpu::Buffer,
    pub scalar_rates: &'a wgpu::Buffer,
    pub paired_delta: &'a wgpu::Buffer,
    pub scalar_delta: &'a wgpu::Buffer,
    pub status: &'a wgpu::Buffer,
}

/// Experimental force-to-rate graph for one immutable unrestricted model snapshot.
///
/// Encodes mass/bias, acceleration and rate publication. Drive contains paired
/// generalized efforts after control evaluation; external loads are world COM
/// wrenches. Joint friction, velocity caps, controls, coordinate drift and body
/// velocity reconstruction are not performed here. Bind outputs to the solver's
/// prepared rate/delta buffers, then perform drift before consuming body state.
/// Prediction consumes paired mass entries; coupling can borrow their f32 high mirror.
pub struct PreparedGeneralizedPrediction {
    operator: ExperimentalBiasKernel,
    solve: ExperimentalMassPredictor,
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    dt: f32,
    damping: f32,
}

fn source(dofs: u32, dt: f32, damping: f32) -> String {
    include_str!("../shaders/generalized_rate_prediction.wgsl")
        .replace("__DOFS__", &dofs.to_string())
        .replace("__STEP__", &dt.to_bits().to_string())
        .replace("__DAMPING__", &damping.to_bits().to_string())
}

fn active_source(text: String) -> String {
    let guard = "    if atomicLoad(&status)!=0u { return; }";
    assert_eq!(text.matches(guard).count(), 1);
    format!("@group(0) @binding(8) var<storage, read> activation_mask: array<u32>;\n{}",
        text.replace(guard,&format!("{guard}\n    if activation_mask[0]==0u {{ return; }}\n    if activation_mask[0]!=1u {{ atomicOr(&status,32u); return; }}")))
}

fn validate_rates(model: &Model) -> Result<()> {
    let caps = model.joint_velocity_limit.host()?;
    let friction = model.joint_friction.host()?;
    if caps.len() != model.joint_dof_total
        || friction.len() != model.joint_dof_total
        || caps.iter().any(|&v| v != f64::INFINITY)
        || friction.iter().any(|&v| v != 0.0)
    {
        return Err(Error::Gpu(
            "generalized prediction requires uncapped frictionless DOFs".into(),
        ));
    }
    Ok(())
}

impl PreparedGeneralizedPrediction {
    /// Buffer budget excludes caller-owned state, drive, output and status buffers.
    pub fn required_bytes(bodies: usize, dofs: usize) -> Result<u64> {
        if !(1..=64).contains(&bodies) || !(1..=32).contains(&dofs) {
            return Err(Error::Gpu(
                "unsupported generalized prediction dimensions".into(),
            ));
        }
        Ok((192 * bodies + 12 * dofs * dofs + 24 * bodies * dofs + 16 * dofs + 8) as u64)
    }

    /// Retains bindings for a fixed positive f32 timestep and nonnegative damping.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        inputs: ExperimentalBiasBindings<'_>,
        drive: &wgpu::Buffer,
        outputs: GeneralizedPredictionOutputs<'_>,
        dt: f64,
        damping: f32,
        budget: u64,
    ) -> Result<Self> {
        Self::build(
            device, queue, model, inputs, drive, outputs, dt, damping, budget, None,
        )
    }

    /// Suppresses publication for an inactive articulation; caller seeds output rates first.
    pub fn new_with_active(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        inputs: ExperimentalBiasBindings<'_>,
        drive: &wgpu::Buffer,
        outputs: GeneralizedPredictionOutputs<'_>,
        dt: f64,
        damping: f32,
        budget: u64,
        active: &wgpu::Buffer,
    ) -> Result<Self> {
        Self::build(
            device,
            queue,
            model,
            inputs,
            drive,
            outputs,
            dt,
            damping,
            budget,
            Some(active),
        )
    }

    fn build(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        inputs: ExperimentalBiasBindings<'_>,
        drive: &wgpu::Buffer,
        outputs: GeneralizedPredictionOutputs<'_>,
        dt: f64,
        damping: f32,
        budget: u64,
        active: Option<&wgpu::Buffer>,
    ) -> Result<Self> {
        let invalid =
            || Error::Gpu("invalid generalized prediction configuration or bindings".into());
        let required = Self::required_bytes(model.body_count, model.joint_dof_total)?;
        let dt = dt as f32;
        let d = model.joint_dof_total as u64;
        validate_rates(model)?;
        if !dt.is_finite()
            || dt <= 0.0
            || !damping.is_finite()
            || damping < 0.0
            || budget < required
        {
            return Err(invalid());
        }
        let out = [
            outputs.paired_rates,
            outputs.scalar_rates,
            outputs.paired_delta,
            outputs.scalar_delta,
            outputs.status,
        ];
        let input = [
            inputs.poses,
            inputs.position_low,
            inputs.paired_rates,
            inputs.external_com_wrenches,
            drive,
        ];
        let limits = device.limits();
        let sizes = [8 * d, 4 * d, 8 * d, 4 * d, 4];
        let binding_count = 8 + u32::from(active.is_some());
        if limits.max_storage_buffers_per_shader_stage < binding_count
            || limits.max_bindings_per_bind_group < binding_count
            || active.is_some_and(|b| {
                b.size() < 4 || !b.usage().contains(wgpu::BufferUsages::STORAGE) || out.contains(&b)
            })
            || out
                .iter()
                .zip(sizes)
                .enumerate()
                .any(|(i, (buffer, size))| {
                    buffer.size() < size
                        || size > limits.max_storage_buffer_binding_size
                        || size > limits.max_buffer_size
                        || !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
                        || input.contains(buffer)
                        || out[..i].contains(buffer)
                })
        {
            return Err(invalid());
        }
        let input_rates = inputs.paired_rates;
        let operator = ExperimentalBiasKernel::new_with_coupling_operators(
            device,
            queue,
            model,
            inputs,
            required - (8 * d + 4),
        )?;
        let solve = ExperimentalMassPredictor::new_with_paired_mass_and_bias_status(
            device,
            d as u32,
            operator.paired_mass_matrix().unwrap(),
            drive,
            operator.bias(),
            operator.status(),
            8 * d + 4,
        )?;
        let mut buffers = vec![
            input_rates,
            solve.acceleration(),
            solve.status(),
            out[0],
            out[1],
            out[2],
            out[3],
            out[4],
        ];
        let mut sizes = vec![8 * d, 8 * d, 4, 8 * d, 4 * d, 8 * d, 4 * d, 4];
        if let Some(active) = active {
            buffers.push(active);
            sizes.push(4);
        }
        let entries: Vec<_> = sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: i < 3 || i == 8,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: std::num::NonZeroU64::new(size),
                },
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("generalized rate prediction"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("generalized rate prediction"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let mut text = source(d as u32, dt, damping);
        if active.is_some() {
            text = active_source(text);
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("generalized rate prediction"),
            source: wgpu::ShaderSource::Wgsl(text.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("generalized rate prediction"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("publish"),
            compilation_options: Default::default(),
            cache: None,
        });
        let entries: Vec<_> = buffers
            .iter()
            .zip(sizes)
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
            label: Some("generalized rate prediction"),
            layout: &layout,
            entries: &entries,
        });
        Ok(Self {
            operator,
            solve,
            pipeline,
            group,
            dt,
            damping,
        })
    }

    /// Encodes three retained dispatches; all published rates/deltas stay intact on failure.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.operator.encode(encoder);
        self.solve.encode(encoder);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }

    /// Clears internal failures only; the caller retains ownership of shared solver status.
    pub fn reset_internal(&self, encoder: &mut wgpu::CommandEncoder) {
        self.operator.reset(encoder);
        self.solve.reset(encoder);
    }

    pub fn timestep(&self) -> f32 {
        self.dt
    }

    pub fn validate_model(&self, model: &Model, dt: f64, damping: f32) -> Result<()> {
        validate_rates(model)?;
        if (dt as f32).to_bits() != self.dt.to_bits() || damping.to_bits() != self.damping.to_bits()
        {
            return Err(Error::Gpu(
                "prepared prediction timestep or damping changed".into(),
            ));
        }
        self.operator.validate_model(model)
    }
    pub fn mass_matrix(&self) -> &wgpu::Buffer {
        self.operator.mass_matrix()
    }
    pub fn paired_mass_matrix(&self) -> &wgpu::Buffer {
        self.operator.paired_mass_matrix().unwrap()
    }

    /// Global-body, row-major 6-by-DOF world-axis COM Jacobians.
    pub fn com_jacobians(&self) -> &wgpu::Buffer {
        self.operator.com_jacobians().unwrap()
    }
    pub fn bias(&self) -> &wgpu::Buffer {
        self.operator.bias()
    }
    pub fn acceleration(&self) -> &wgpu::Buffer {
        self.solve.acceleration()
    }
}

#[cfg(test)]
#[path = "generalized_prediction_tests.rs"]
mod tests;
