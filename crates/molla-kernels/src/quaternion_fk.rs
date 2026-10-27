//! Retained high/low FK orientations derived from device joint coordinates.

use molla_compute::Engine;
use molla_core::{Error, Result};
use molla_sim::Model;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuQuaternionPair {
    pub high: [f32; 4],
    pub low: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Joint {
    metadata: [u32; 4],
    axis: [f32; 4],
    axis_low: [f32; 4],
    parent: GpuQuaternionPair,
    child: GpuQuaternionPair,
}

pub struct QuaternionFkBindings<'a> {
    pub coordinates: Option<&'a wgpu::Buffer>,
    pub coordinate_residues: Option<&'a wgpu::Buffer>,
    pub poses: &'a wgpu::Buffer,
    pub rotations: &'a wgpu::Buffer,
    pub rounding: &'a wgpu::Buffer,
    pub status: &'a wgpu::Buffer,
}

pub struct QuaternionFkKernel {
    initialize: wgpu::ComputePipeline,
    evaluate: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    joints: wgpu::Buffer,
    starts: wgpu::Buffer,
    zero: wgpu::Buffer,
    bodies: u32,
    articulations: u32,
    coordinates: u64,
}

pub struct QuaternionFkDispatch {
    initialize: wgpu::ComputePipeline,
    evaluate: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    bodies: u32,
    articulations: u32,
}

impl QuaternionFkDispatch {
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        for (pipeline, count) in [
            (&self.initialize, self.bodies),
            (&self.evaluate, self.articulations),
        ] {
            if count == 0 {
                continue;
            }
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.group, &[]);
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
    }
}

impl QuaternionFkKernel {
    pub fn new(engine: &Engine, model: &Model) -> Result<Self> {
        Self::build(engine, model, false)
    }

    /// Retains quaternion tails produced by compensated coordinate integration.
    pub fn new_with_coordinate_residues(engine: &Engine, model: &Model) -> Result<Self> {
        Self::build(engine, model, true)
    }

    /// Retains quaternion tails produced by compensated coordinate integration.
    pub fn new_with_free_coordinate_residues(engine: &Engine, model: &Model) -> Result<Self> {
        Self::new_with_coordinate_residues(engine, model)
    }

    fn build(engine: &Engine, model: &Model, coordinate_residues: bool) -> Result<Self> {
        let invalid = || Error::Gpu("invalid quaternion FK model".into());
        let device = engine.device().ok_or_else(invalid)?;
        let queue = engine.queue().ok_or_else(invalid)?;
        let kinds = model.joint_type.host()?;
        let parents = model.joint_parent.host()?;
        let children = model.joint_child.host()?;
        let offsets = model.joint_coord_offset.host()?;
        let axes = model.joint_axis.host()?;
        let reversed = model.joint_reversed.host()?;
        let xp = model.joint_x_p.host()?;
        let xc = model.joint_x_c.host()?;
        let flags = model.body_flags.host()?;
        let starts = model.articulation_start.host()?;
        let bodies = u32::try_from(model.body_count).map_err(|_| invalid())?;
        let articulations = u32::try_from(model.articulation_count).map_err(|_| invalid())?;
        if [
            kinds.len(),
            parents.len(),
            children.len(),
            axes.len(),
            xp.len(),
            xc.len(),
            reversed.len(),
        ]
        .iter()
        .any(|&len| len != model.joint_count)
            || bodies > i32::MAX as u32
            || offsets.len() != model.joint_count + 1
            || offsets.first() != Some(&0)
            || offsets.last().copied().map(|n| n as usize) != Some(model.joint_coord_total)
            || flags.len() != model.body_count
            || starts.len() != model.articulation_count + 1
            || starts.first() != Some(&0)
            || starts.last().copied().map(|n| n as usize) != Some(model.joint_count)
            || starts.windows(2).any(|s| s[0] > s[1])
            || [bodies, articulations]
                .iter()
                .any(|n| n.div_ceil(64) > device.limits().max_compute_workgroups_per_dimension)
        {
            return Err(invalid());
        }
        let mut owners = vec![None; model.body_count];
        for (art, range) in starts.windows(2).enumerate() {
            for index in range[0] as usize..range[1] as usize {
                let child = children[index] as usize;
                if child >= model.body_count || owners[child].replace((art, index)).is_some() {
                    return Err(invalid());
                }
            }
        }
        let pair = |values: [f64; 4]| -> Result<GpuQuaternionPair> {
            if values.iter().any(|v| !v.is_finite())
                || (values.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() > 1e-6
            {
                return Err(invalid());
            }
            let high = values.map(|v| v as f32);
            Ok(GpuQuaternionPair {
                high,
                low: std::array::from_fn(|i| (values[i] - f64::from(high[i])) as f32),
            })
        };
        let mut packed = Vec::with_capacity(model.joint_count);
        for index in 0..model.joint_count {
            let child = children[index] as usize;
            let parent = parents[index];
            let kind = kinds[index];
            let widths = [1, 1, 4, 0, 7, 7, 7, 2];
            if kind > 7
                || reversed[index] > 1
                || parent < -1
                || parent >= bodies as i32
                || parent == child as i32
                || offsets[index].checked_add(widths[kind as usize]) != Some(offsets[index + 1])
                || !axes[index].is_finite()
                || (matches!(kind, 0 | 1) && (axes[index].length_squared() - 1.0).abs() > 1e-6)
            {
                return Err(invalid());
            }
            if parent >= 0
                && let Some((art, prior)) = owners[parent as usize]
            {
                let (child_art, _) = owners[child].unwrap();
                if flags[parent as usize] & 3 == 0 && (art != child_art || prior >= index) {
                    return Err(invalid());
                }
            }
            let high = axes[index].to_array().map(|v| v as f32);
            packed.push(Joint {
                metadata: [parent as u32, child as u32, kind, offsets[index]],
                axis: [high[0], high[1], high[2], (flags[child] & 3) as f32],
                axis_low: std::array::from_fn(|i| {
                    if i < 3 {
                        (axes[index][i] - f64::from(high[i])) as f32
                    } else {
                        (u32::from(coordinate_residues && matches!(kind, 2 | 4..=6))
                            | (reversed[index] << 1)) as f32
                    }
                }),
                parent: pair(xp[index].rotation.to_array())?,
                child: pair(xc[index].rotation.to_array())?,
            });
        }
        let storage = |bytes: &[u8], minimum: u64| -> Result<wgpu::Buffer> {
            let size = (bytes.len() as u64).max(minimum);
            if size > device.limits().max_storage_buffer_binding_size {
                return Err(invalid());
            }
            let buffer = molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("quaternion FK model"),
                    size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }
            );
            if !bytes.is_empty() {
                molla_core::gpu_write_buffer!(queue, &buffer, 0, bytes);
            }
            Ok(buffer)
        };
        let joints = storage(bytemuck::cast_slice(&packed), 112)?;
        let starts = storage(bytemuck::cast_slice(starts), 4)?;
        let zero = storage(&[0; 4], 4)?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("quaternion FK"),
            source: wgpu::ShaderSource::Wgsl(
                concat!(
                    include_str!("../shaders/quaternion_precision.wgsl"),
                    include_str!("../shaders/quaternion_fk.wgsl")
                )
                .into(),
            ),
        });
        let entries: Vec<_> = (0..8)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                count: None,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: binding < 5,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("quaternion FK"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("quaternion FK"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Ok(Self {
            initialize: pipeline("initialize"),
            evaluate: pipeline("evaluate"),
            layout,
            joints,
            starts,
            zero,
            bodies,
            articulations,
            coordinates: model.joint_coord_total as u64,
        })
    }

    pub fn prepare(
        &self,
        device: &wgpu::Device,
        bindings: QuaternionFkBindings<'_>,
    ) -> Result<QuaternionFkDispatch> {
        let coordinates = bindings.coordinates.unwrap_or(&self.zero);
        let residues = bindings.coordinate_residues.unwrap_or(&self.zero);
        let buffers = [
            &self.joints,
            &self.starts,
            coordinates,
            residues,
            bindings.poses,
            bindings.rotations,
            bindings.rounding,
            bindings.status,
        ];
        if (self.coordinates != 0
            && (bindings.coordinates.is_none() || bindings.coordinate_residues.is_none()))
            || coordinates.size() < self.coordinates.max(1) * 4
            || residues.size() < self.coordinates.max(1) * 4
            || self.bodies == 0
            || bindings.poses.size() != u64::from(self.bodies) * 32
            || bindings.rotations.size() != bindings.poses.size()
            || bindings.rounding.size() < u64::from(self.bodies) * 4
            || bindings.status.size() < 4
            || buffers.iter().any(|buffer| {
                !buffer.usage().contains(wgpu::BufferUsages::STORAGE)
                    || buffer.size() > device.limits().max_storage_buffer_binding_size
            })
        {
            return Err(Error::Gpu("invalid quaternion FK capacity".into()));
        }
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("quaternion FK"),
            layout: &self.layout,
            entries: &entries,
        });
        Ok(QuaternionFkDispatch {
            initialize: self.initialize.clone(),
            evaluate: self.evaluate.clone(),
            group,
            bodies: self.bodies,
            articulations: self.articulations,
        })
    }
}

const _: () = {
    assert!(std::mem::size_of::<GpuQuaternionPair>() == 32);
    assert!(std::mem::size_of::<Joint>() == 112);
};
