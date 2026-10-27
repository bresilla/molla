//! Retained GJK/EPA and analytic primitive contacts with device counts and sticky faults.

use molla_core::{Error, Result};
use molla_sim::Model;

pub const NARROW_FAULT_CAPACITY: u32 = 1;
pub const NARROW_FAULT_INDEX: u32 = 2;
pub const NARROW_FAULT_GJK: u32 = 4;
pub const NARROW_FAULT_EPA: u32 = 8;
pub const NARROW_FAULT_NONFINITE: u32 = 16;
pub const RESIDENT_MANIFOLD_POINTS: u32 = 12;

#[derive(Clone, Copy, Debug, Default)]
pub struct ResidentCollisionOptions<'a> {
    pub include_sensor_overlaps: bool,
    pub extra_shape_exclusions: &'a [(u32, u32)],
    /// Optional per-pose vec4<f32> body translation residues for analytic contact depths.
    /// Convex GJK/EPA and published world witnesses use the high pose components.
    pub plane_position_residues: &'a [&'a wgpu::Buffer],
}

/// Fixed-capacity device narrow phase. Recreate after any bound buffer changes.
/// Unsupported shapes and GJK/EPA nonconvergence are faults, never CPU fallbacks.
pub struct ResidentNarrowPhase {
    gjk: wgpu::ComputePipeline,
    epa: wgpu::ComputePipeline,
    shapes: Vec<wgpu::BindGroup>,
    pairs: [wgpu::BindGroup; 2],
    outputs: [wgpu::BindGroup; 2],
    control: wgpu::BindGroup,
    overlap: wgpu::Buffer,
    simplex: wgpu::Buffer,
    contacts: wgpu::Buffer,
    manifold_points: wgpu::Buffer,
    manifold_counts: wgpu::Buffer,
    status: wgpu::Buffer,
    capacity: u32,
}

impl ResidentNarrowPhase {
    /// Bind uploaded model geometry, pose alternatives and device pair buffers.
    /// `count[0]` is the emitted count, including any broad-phase overflow.
    /// Uses model collision permissions and excludes sensor overlaps by default.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        model: &Model,
        poses: &[&wgpu::Buffer],
        pair_a: &wgpu::Buffer,
        pair_b: &wgpu::Buffer,
        count: &wgpu::Buffer,
        capacity: u32,
    ) -> Result<Self> {
        Self::with_collision_options(
            device,
            model,
            poses,
            pair_a,
            pair_b,
            count,
            capacity,
            ResidentCollisionOptions::default(),
        )
    }

    /// Compile model permissions and optional additional exclusions into retained buffers.
    /// Rebuild after flags, worlds, masks, membership or exclusion edits.
    #[allow(clippy::too_many_arguments)]
    pub fn with_collision_options(
        device: &wgpu::Device,
        model: &Model,
        poses: &[&wgpu::Buffer],
        pair_a: &wgpu::Buffer,
        pair_b: &wgpu::Buffer,
        count: &wgpu::Buffer,
        capacity: u32,
        options: ResidentCollisionOptions<'_>,
    ) -> Result<Self> {
        let limits = device.limits();
        if capacity == 0
            || capacity.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || u64::from(capacity) * 64 * u64::from(RESIDENT_MANIFOLD_POINTS)
                > limits.max_storage_buffer_binding_size
            || limits.max_storage_buffers_per_shader_stage < 17
            || model.shape_count == 0
            || model.shape_count > u32::MAX as usize
            || poses.is_empty()
        {
            return Err(Error::Gpu(
                "invalid resident narrow-phase capacity or model".into(),
            ));
        }
        let geometry = [
            checked(
                model.shape_geom_kind.device_buffer(),
                model.shape_count as u64 * 4,
            )?,
            checked(
                model.shape_geom_data.device_buffer(),
                model.shape_count as u64 * 16,
            )?,
            checked(
                model.shape_body.device_buffer(),
                model.shape_count as u64 * 4,
            )?,
            checked(
                model.shape_transform.device_buffer(),
                model.shape_count as u64 * 32,
            )?,
        ];
        for pose in poses {
            checked(Some(pose), (model.body_count.max(1) as u64) * 32)?;
        }
        if !options.plane_position_residues.is_empty()
            && options.plane_position_residues.len() != poses.len()
        {
            return Err(Error::Gpu(
                "invalid plane position residue state count".into(),
            ));
        }
        for residue in options.plane_position_residues {
            checked(Some(residue), model.body_count.max(1) as u64 * 16)?;
        }
        checked(Some(pair_a), u64::from(capacity) * 4)?;
        checked(Some(pair_b), u64::from(capacity) * 4)?;
        checked(Some(count), 4)?;
        let (permissions, exclusions) = collision_data(model, options)?;
        if permissions.len() as u64 * 16 > limits.max_storage_buffer_binding_size
            || exclusions.len().max(1) as u64 * 8 > limits.max_storage_buffer_binding_size
        {
            return Err(Error::Gpu(
                "resident collision filter exceeds storage limits".into(),
            ));
        }

        let shape_layout = storage_layout(device, "resident shapes", 6, true);
        let gjk_pairs = storage_layout(device, "resident GJK pairs", 2, true);
        let epa_pairs = storage_layout(device, "resident EPA pairs", 4, true);
        let gjk_outputs = storage_layout(device, "resident GJK outputs", 2, false);
        let epa_outputs = storage_layout(device, "resident EPA outputs", 3, false);
        let control_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("resident narrow-phase control"),
            entries: &[
                layout_entry(0, wgpu::BufferBindingType::Uniform),
                layout_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                layout_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
                layout_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                layout_entry(4, wgpu::BufferBindingType::Storage { read_only: true }),
            ],
        });
        let gjk = pipeline(
            device,
            "resident_gjk",
            &[&shape_layout, &gjk_pairs, &gjk_outputs, &control_layout],
            &format!(
                "{}\n{}\n{}",
                include_str!("../shaders/gjk.wgsl"),
                include_str!("../shaders/narrow_common.wgsl"),
                include_str!("../shaders/narrow_gjk.wgsl")
            ),
        );
        let epa = pipeline(
            device,
            "resident_epa",
            &[&shape_layout, &epa_pairs, &epa_outputs, &control_layout],
            &format!(
                "{}\n{}\n{}",
                include_str!("../shaders/epa.wgsl"),
                include_str!("../shaders/narrow_common.wgsl"),
                include_str!("../shaders/narrow_epa.wgsl")
            ),
        );

        let overlap = storage(device, "resident overlap", u64::from(capacity) * 4);
        let simplex = storage(device, "resident simplex", u64::from(capacity) * 192);
        let contacts = storage(device, "resident contacts", u64::from(capacity) * 64);
        let manifold_points = storage(
            device,
            "resident manifold points",
            u64::from(capacity) * 64 * u64::from(RESIDENT_MANIFOLD_POINTS),
        );
        let manifold_counts = storage(device, "resident manifold counts", u64::from(capacity) * 4);
        let status = storage(device, "resident narrow-phase status", 4);
        let uniform = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("resident narrow-phase uniform"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: true,
            }
        );
        molla_core::gpu_mapped_upload!(uniform.slice(..)).copy_from_slice(bytemuck::cast_slice(&[
            capacity,
            model.shape_count as u32,
            exclusions.len() as u32,
            u32::from(options.include_sensor_overlaps),
        ]));
        uniform.unmap();
        let permission_buffer = initialized(
            device,
            "resident collision permissions",
            bytemuck::cast_slice(&permissions),
        );
        let exclusion_buffer = initialized(
            device,
            "resident collision exclusions",
            if exclusions.is_empty() {
                &[0; 8]
            } else {
                bytemuck::cast_slice(&exclusions)
            },
        );
        let zero_residues = storage(
            device,
            "zero plane position residues",
            model.body_count.max(1) as u64 * 16,
        );
        let shapes = poses
            .iter()
            .enumerate()
            .map(|(index, pose)| {
                group(
                    device,
                    &shape_layout,
                    &[
                        geometry[0],
                        geometry[1],
                        geometry[2],
                        geometry[3],
                        pose,
                        options
                            .plane_position_residues
                            .get(index)
                            .copied()
                            .unwrap_or(&zero_residues),
                    ],
                )
            })
            .collect();
        let pairs = [
            group(device, &gjk_pairs, &[pair_a, pair_b]),
            group(device, &epa_pairs, &[pair_a, pair_b, &overlap, &simplex]),
        ];
        let outputs = [
            group(device, &gjk_outputs, &[&overlap, &simplex]),
            group(
                device,
                &epa_outputs,
                &[&contacts, &manifold_points, &manifold_counts],
            ),
        ];
        let control = group(
            device,
            &control_layout,
            &[
                &uniform,
                count,
                &status,
                &permission_buffer,
                &exclusion_buffer,
            ],
        );
        Ok(Self {
            gjk,
            epa,
            shapes,
            pairs,
            outputs,
            control,
            overlap,
            simplex,
            contacts,
            manifold_points,
            manifold_counts,
            status,
            capacity,
        })
    }

    /// Append two passes without resource creation, submission, uploads or readback.
    /// Pre-existing faults halt evaluation and retire scratch until `encode_reset_status`.
    /// Trust outputs only at status zero.
    /// Inactive output slots are cleared, while simplex slots require overlap == 1.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, pose: usize) -> Result<()> {
        let shapes = self
            .shapes
            .get(pose)
            .ok_or_else(|| Error::Gpu("invalid narrow-phase pose binding".into()))?;
        for (index, pipeline) in [&self.gjk, &self.epa].into_iter().enumerate() {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("resident narrow phase"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, shapes, &[]);
            pass.set_bind_group(1, &self.pairs[index], &[]);
            pass.set_bind_group(2, &self.outputs[index], &[]);
            pass.set_bind_group(3, &self.control, &[]);
            pass.dispatch_workgroups(self.capacity.div_ceil(64), 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
        Ok(())
    }

    pub fn encode_reset_status(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.status, 0, None);
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }
    pub fn status_buffer(&self) -> &wgpu::Buffer {
        &self.status
    }
    /// First contact per pair: normal/depth, point A, point B, status bits; 64 bytes.
    pub fn contact_buffer(&self) -> &wgpu::Buffer {
        &self.contacts
    }
    /// Twelve 64-byte contact slots per pair, in the same layout as `contact_buffer`.
    /// Plane pairs have up to twelve points; other convex pairs have at most one.
    pub fn manifold_buffer(&self) -> &wgpu::Buffer {
        &self.manifold_points
    }
    /// Number of valid manifold slots per pair. Unused slots are zeroed each dispatch.
    pub fn manifold_count_buffer(&self) -> &wgpu::Buffer {
        &self.manifold_counts
    }
    /// 0: skipped/disjoint, 1: convex overlap, 2: fault, 3: plane, 4: sphere-box, 5: sphere-sphere, 6: capsule-capsule.
    pub fn overlap_buffer(&self) -> &wgpu::Buffer {
        &self.overlap
    }
    pub fn simplex_buffer(&self) -> &wgpu::Buffer {
        &self.simplex
    }
    /// Owned buffers, retained bind groups and compute pipelines, excluding caller inputs.
    pub fn resource_counts(&self) -> [usize; 3] {
        [9, self.shapes.len() + 5, 2]
    }
}

fn collision_data(
    model: &Model,
    options: ResidentCollisionOptions<'_>,
) -> Result<(Vec<[u32; 4]>, Vec<[u32; 2]>)> {
    let bodies = model.shape_body.host()?;
    let worlds = model.body_world.host()?;
    let flags = model.shape_flags.host()?;
    let groups = model.shape_collision_group.host()?;
    let masks = model.shape_collision_mask.host()?;
    if [bodies.len(), flags.len(), groups.len(), masks.len()]
        .iter()
        .any(|&n| n != model.shape_count)
    {
        return Err(Error::Gpu(
            "invalid resident collision filter table length".into(),
        ));
    }
    let mut permissions = Vec::with_capacity(model.shape_count);
    for i in 0..model.shape_count {
        let world = if bodies[i] == -1 {
            -1
        } else if bodies[i] >= 0 {
            *worlds
                .get(bodies[i] as usize)
                .ok_or_else(|| Error::Gpu("invalid collision filter body".into()))?
        } else {
            return Err(Error::Gpu("invalid collision filter body".into()));
        };
        permissions.push([flags[i], groups[i], masks[i], world as u32]);
    }
    let mut exclusions = Vec::new();
    for &(a, b) in model
        .shape_collision_filter_pairs
        .iter()
        .chain(options.extra_shape_exclusions)
    {
        if a as usize >= model.shape_count || b as usize >= model.shape_count || a == b {
            return Err(Error::Gpu("invalid resident collision exclusion".into()));
        }
        exclusions.push([a.min(b), a.max(b)]);
    }
    exclusions.sort_unstable();
    exclusions.dedup();
    Ok((permissions, exclusions))
}

fn initialized(device: &wgpu::Device, label: &str, bytes: &[u8]) -> wgpu::Buffer {
    let buffer = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        }
    );
    molla_core::gpu_mapped_upload!(buffer.slice(..)).copy_from_slice(bytes);
    buffer.unmap();
    buffer
}

fn checked(buffer: Option<&wgpu::Buffer>, bytes: u64) -> Result<&wgpu::Buffer> {
    buffer
        .filter(|b| b.size() >= bytes && b.usage().contains(wgpu::BufferUsages::STORAGE))
        .ok_or_else(|| Error::Gpu("missing or undersized narrow-phase storage buffer".into()))
}

fn storage(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }
    )
}

fn layout_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_layout(
    device: &wgpu::Device,
    label: &str,
    count: u32,
    read_only: bool,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &(0..count)
            .map(|i| layout_entry(i, wgpu::BufferBindingType::Storage { read_only }))
            .collect::<Vec<_>>(),
    })
}

fn group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    buffers: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("resident narrow phase"),
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

fn pipeline(
    device: &wgpu::Device,
    entry: &str,
    layouts: &[&wgpu::BindGroupLayout],
    source: &str,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(entry),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(entry),
        bind_group_layouts: &layouts.iter().copied().map(Some).collect::<Vec<_>>(),
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(&layout),
        module: &module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}
