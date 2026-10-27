//! Retained normal-contact COM wrenches and material/slip rows for device manifolds.

use molla_core::{Error, Result};
use molla_geometry::RESIDENT_MANIFOLD_POINTS;
use molla_sim::{CoefficientCombineRule, FrictionCoefficients, Model};

pub const CONTACT_FAULT_INPUT: u32 = 32;
pub const CONTACT_FAULT_NONFINITE: u32 = 64;
pub const CONTACT_FAULT_MATERIAL: u32 = 128;

pub struct ContactStateBuffers<'a> {
    pub poses: &'a wgpu::Buffer,
    pub velocities: &'a wgpu::Buffer,
    pub external_forces: &'a wgpu::Buffer,
}

/// One 128-byte row per manifold slot; forces are normal-only, in world coordinates.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct NormalContactRow {
    pub point_depth: [f32; 4],
    pub normal_force: [f32; 4],
    /// Static friction, dynamic friction, restitution and response eligibility, including zero depth.
    pub coefficients: [f32; 4],
    /// Shape A, shape B, body A and body B; missing bodies use u32::MAX.
    pub endpoints: [u32; 4],
    pub relative_velocity: [f32; 4],
    pub surface_velocity: [f32; 4],
    pub force_on_b: [f32; 4],
    /// Spring stiffness, compression damping, viscous gain and smoothing speed.
    pub penalty: [f32; 4],
}

/// Prepared normal forces; no tangential impulse or restitution impulse is applied.
/// Rebuild after model/material edits or any bound buffer replacement.
pub struct ResidentNormalContacts {
    pipelines: [wgpu::ComputePipeline; 3],
    geometry: wgpu::BindGroup,
    states: Vec<wgpu::BindGroup>,
    outputs: wgpu::BindGroup,
    control: wgpu::BindGroup,
    rows: wgpu::Buffer,
    forces: wgpu::Buffer,
    heads: wgpu::Buffer,
    links: wgpu::Buffer,
    body_count: u32,
    row_count: u32,
}

impl ResidentNormalContacts {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        model: &Model,
        states: &[ContactStateBuffers<'_>],
        pair_a: &wgpu::Buffer,
        pair_b: &wgpu::Buffer,
        pair_count: &wgpu::Buffer,
        manifold_counts: &wgpu::Buffer,
        manifold_points: &wgpu::Buffer,
        status: &wgpu::Buffer,
        capacity: u32,
    ) -> Result<Self> {
        let limits = device.limits();
        let row_count = capacity
            .checked_mul(RESIDENT_MANIFOLD_POINTS)
            .ok_or_else(|| Error::Gpu("contact row capacity overflow".into()))?;
        if capacity == 0
            || states.is_empty()
            || model.shape_count == 0
            || model.shape_count > u32::MAX as usize
            || model.body_count > i32::MAX as usize
            || row_count > u32::MAX / 2
            || row_count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || (model.body_count as u64).div_ceil(64)
                > u64::from(limits.max_compute_workgroups_per_dimension)
            || limits.max_storage_buffers_per_shader_stage < 15
        {
            return Err(Error::Gpu(
                "invalid normal-contact capacity or model".into(),
            ));
        }
        for bytes in [
            u64::from(row_count) * 128,
            model.body_count.max(1) as u64 * 32,
            model.shape_count as u64 * 64,
        ] {
            if bytes > limits.max_storage_buffer_binding_size {
                return Err(Error::Gpu("normal-contact storage limit exceeded".into()));
            }
        }
        let check = |buffer: &wgpu::Buffer, bytes| -> Result<()> {
            if buffer.size() < bytes || !buffer.usage().contains(wgpu::BufferUsages::STORAGE) {
                return Err(Error::Gpu(
                    "missing or undersized normal-contact buffer".into(),
                ));
            }
            Ok(())
        };
        for buffer in [pair_a, pair_b, manifold_counts] {
            check(buffer, u64::from(capacity) * 4)?;
        }
        check(pair_count, 4)?;
        check(status, 4)?;
        check(manifold_points, u64::from(row_count) * 64)?;
        for state in states {
            for buffer in [state.poses, state.velocities, state.external_forces] {
                check(buffer, model.body_count.max(1) as u64 * 32)?;
            }
        }
        let (shape_data, body_data) = metadata(model)?;
        let shapes = initialized(
            device,
            "contact shape metadata",
            bytemuck::cast_slice(&shape_data),
            false,
        );
        let bodies = initialized(
            device,
            "contact body metadata",
            bytemuck::cast_slice(&body_data),
            false,
        );
        let uniform = initialized(
            device,
            "contact uniform",
            bytemuck::cast_slice(&[
                capacity,
                model.body_count as u32,
                model.shape_count as u32,
                RESIDENT_MANIFOLD_POINTS,
            ]),
            true,
        );
        let rows = storage(device, "normal contact rows", u64::from(row_count) * 128);
        let forces = storage(
            device,
            "normal contact body forces",
            model.body_count.max(1) as u64 * 32,
        );
        let heads = storage(
            device,
            "normal contact body heads",
            model.body_count.max(1) as u64 * 4,
        );
        let links = storage(device, "normal contact links", u64::from(row_count) * 8);
        let geometry_layout = layout(device, &[true; 5], false);
        let state_layout = layout(device, &[true; 5], false);
        let output_layout = layout(device, &[false; 4], false);
        let control_layout = layout(device, &[true, false], true);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("resident normal contacts"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/normal_contact.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("resident normal contacts"),
            bind_group_layouts: &[
                Some(&geometry_layout),
                Some(&state_layout),
                Some(&output_layout),
                Some(&control_layout),
            ],
            immediate_size: 0,
        });
        let pipelines = ["reset_heads", "normal_contacts", "gather_forces"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let geometry = group(
            device,
            &geometry_layout,
            &[pair_a, pair_b, pair_count, manifold_counts, manifold_points],
        );
        let states = states
            .iter()
            .map(|state| {
                group(
                    device,
                    &state_layout,
                    &[
                        &shapes,
                        &bodies,
                        state.poses,
                        state.velocities,
                        state.external_forces,
                    ],
                )
            })
            .collect();
        let outputs = group(device, &output_layout, &[&rows, &heads, &links, &forces]);
        let control = group(device, &control_layout, &[&uniform, status]);
        Ok(Self {
            pipelines,
            geometry,
            states,
            outputs,
            control,
            rows,
            forces,
            heads,
            links,
            body_count: model.body_count as u32,
            row_count,
        })
    }

    /// Record three passes without allocating, uploading, submitting or mapping.
    /// Caller owns the shared sticky status; outputs are valid only at status zero.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, state: usize) -> Result<()> {
        self.encode_phases(encoder, state, 0..3)
    }

    /// Prepare normal rows and body incidence before optional row-level corrections.
    pub fn encode_rows(&self, encoder: &mut wgpu::CommandEncoder, state: usize) -> Result<()> {
        self.encode_phases(encoder, state, 0..2)
    }

    /// Gather the current row forces after preparation and any onset suppression.
    pub fn encode_forces(&self, encoder: &mut wgpu::CommandEncoder, state: usize) -> Result<()> {
        self.encode_phases(encoder, state, 2..3)
    }

    fn encode_phases(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        state: usize,
        phases: std::ops::Range<usize>,
    ) -> Result<()> {
        let state = self
            .states
            .get(state)
            .ok_or_else(|| Error::Gpu("invalid normal-contact state binding".into()))?;
        for phase in phases {
            let count = if phase == 1 {
                self.row_count
            } else {
                self.body_count.max(1)
            };
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("resident normal contacts"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines[phase]);
            pass.set_bind_group(0, &self.geometry, &[]);
            pass.set_bind_group(1, state, &[]);
            pass.set_bind_group(2, &self.outputs, &[]);
            pass.set_bind_group(3, &self.control, &[]);
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
        Ok(())
    }

    pub fn row_buffer(&self) -> &wgpu::Buffer {
        &self.rows
    }
    /// World COM wrenches, including the selected state's external force buffer.
    pub fn force_buffer(&self) -> &wgpu::Buffer {
        &self.forces
    }
    pub(crate) fn incidence_buffers(&self) -> (&wgpu::Buffer, &wgpu::Buffer) {
        (&self.heads, &self.links)
    }
    pub fn resource_counts(&self) -> [usize; 3] {
        [7, self.states.len() + 3, 3]
    }
}

fn scalar(value: f64) -> Result<f32> {
    let packed = value as f32;
    if !value.is_finite() || !packed.is_finite() || (value != 0.0 && !packed.is_normal()) {
        return Err(Error::Gpu(
            "contact value is not representable as normal f32".into(),
        ));
    }
    Ok(packed)
}

fn metadata(model: &Model) -> Result<(Vec<[u32; 16]>, Vec<[u32; 8]>)> {
    let body = model.shape_body.host()?;
    let flags = model.shape_flags.host()?;
    let friction = model.shape_friction.host()?;
    let restitution = model.shape_restitution.host()?;
    let com = model.body_com.host()?;
    let body_flags = model.body_flags.host()?;
    if [
        body.len(),
        flags.len(),
        friction.len(),
        restitution.len(),
        model.shape_contact_material.len(),
        model.shape_surface_velocity.len(),
    ]
    .iter()
    .any(|&n| n != model.shape_count)
        || com.len() != model.body_count
        || body_flags.len() != model.body_count
    {
        return Err(Error::Gpu("invalid contact metadata lengths".into()));
    }
    let mut shapes = Vec::with_capacity(model.shape_count);
    for i in 0..model.shape_count {
        if body[i] < -1 || body[i] >= model.body_count as i32 {
            return Err(Error::Gpu("invalid contact shape body".into()));
        }
        let material = model.shape_contact_material[i];
        material.validate()?;
        let mu = material
            .friction
            .unwrap_or(FrictionCoefficients::uniform(friction[i]));
        mu.validate()?;
        let rest = restitution[i];
        if rest < 0.0 {
            return Err(Error::Gpu("invalid contact restitution".into()));
        }
        let mut row = [0; 16];
        for (j, value) in [
            material.stiffness,
            material.damping,
            material.friction_stiffness,
            material.friction_smoothing,
            mu.mu_static,
            mu.mu_dynamic,
            rest,
            0.0,
            model.shape_surface_velocity[i].x,
            model.shape_surface_velocity[i].y,
            model.shape_surface_velocity[i].z,
            0.0,
        ]
        .into_iter()
        .enumerate()
        {
            row[j] = scalar(value)?.to_bits();
        }
        row[12] = body[i] as u32;
        row[13] = flags[i];
        row[14] = match material.friction_combine {
            CoefficientCombineRule::Average => 0,
            CoefficientCombineRule::Min => 1,
            CoefficientCombineRule::Multiply => 2,
            CoefficientCombineRule::Max => 3,
        };
        shapes.push(row);
    }
    let mut bodies = Vec::with_capacity(model.body_count.max(1));
    for i in 0..model.body_count {
        let c = com[i];
        bodies.push([
            scalar(c.x)?.to_bits(),
            scalar(c.y)?.to_bits(),
            scalar(c.z)?.to_bits(),
            0,
            body_flags[i],
            0,
            0,
            0,
        ]);
    }
    if bodies.is_empty() {
        bodies.push([0; 8]);
    }
    Ok((shapes, bodies))
}

fn storage(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }
    )
}

fn initialized(device: &wgpu::Device, label: &str, bytes: &[u8], uniform: bool) -> wgpu::Buffer {
    let buffer = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.len() as u64,
            usage: if uniform {
                wgpu::BufferUsages::UNIFORM
            } else {
                wgpu::BufferUsages::STORAGE
            },
            mapped_at_creation: true,
        }
    );
    molla_core::gpu_mapped_upload!(buffer.slice(..)).copy_from_slice(bytes);
    buffer.unmap();
    buffer
}

fn layout(device: &wgpu::Device, reads: &[bool], uniform: bool) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("normal contact layout"),
        entries: &reads
            .iter()
            .enumerate()
            .map(|(binding, &read_only)| wgpu::BindGroupLayoutEntry {
                binding: binding as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: if uniform && binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            })
            .collect::<Vec<_>>(),
    })
}

fn group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("normal contact bindings"),
        layout,
        entries: &buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    })
}
