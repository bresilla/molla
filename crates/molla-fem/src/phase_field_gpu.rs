//! GPU AT2 phase-field damage Jacobi solve. Mirrors the inner loop of
//! the CPU [`solve_phase_field_fracture_jacobi`](crate::phase_field::solve_phase_field_fracture_jacobi):
//! the linear system is assembled host-side, then the bounded,
//! irreversible Jacobi sweeps run on the device (one thread per row,
//! ping-ponged). Returns the damage after a fixed number of sweeps.

use crate::mesh::FemMesh;
use crate::phase_field::{PhaseFieldFractureParams, assemble_phase_field_fracture_system};
use molla_core::{Error, Result};
use molla_math::Real;
use molla_storage::{Buffer, BufferUsage};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    n: [u32; 4],
}

const WORKGROUP_SIZE: u32 = 64;

/// Run `sweeps` bounded-irreversible Jacobi iterations of the AT2
/// damage solve on the GPU. The result matches the CPU
/// [`solve_phase_field_fracture_jacobi`](crate::phase_field::solve_phase_field_fracture_jacobi)
/// run for the same fixed iteration count within f32 tolerance.
pub fn solve_phase_field_fracture_jacobi_gpu(
    engine: &molla_compute::Engine,
    mesh: &FemMesh,
    damage_old: &[Real],
    history: &[Real],
    params: PhaseFieldFractureParams,
    sweeps: usize,
) -> Result<Vec<Real>> {
    let n = mesh.vert_count();
    if damage_old.len() != n {
        return Err(Error::Fem(format!(
            "phase_field_gpu: damage_old.len() = {} but mesh has {n} vertices",
            damage_old.len()
        )));
    }
    let (lhs, rhs) = assemble_phase_field_fracture_system(mesh, history, params)?;

    if sweeps == 0 || n == 0 {
        return Ok(damage_old.iter().map(|&d| d.clamp(0.0, 1.0)).collect());
    }

    let device = engine
        .device()
        .ok_or_else(|| Error::Gpu("phase_field_gpu: engine has no device".into()))?
        .clone();
    let queue = engine
        .queue()
        .ok_or_else(|| Error::Gpu("phase_field_gpu: engine has no queue".into()))?
        .clone();

    // Matrix (scalar BSR, b = 1) + rhs + lower bound.
    let row_off: Vec<u32> = lhs.row_offsets.iter().map(|&x| x as u32).collect();
    let col_idx: Vec<u32> = lhs.col_idx.iter().map(|&x| x as u32).collect();
    let row_buf = make_storage(&device, "molla::pf row_off", bytemuck::cast_slice(&row_off));
    queue.write_buffer(&row_buf, 0, bytemuck::cast_slice(&row_off));
    let col_buf = make_storage(&device, "molla::pf col_idx", bytemuck::cast_slice(&col_idx));
    queue.write_buffer(&col_buf, 0, bytemuck::cast_slice(&col_idx));

    let mut val_buf = Buffer::from_vec(lhs.values.clone(), BufferUsage::Storage);
    val_buf.ensure_device(&device)?;
    val_buf.upload(&queue)?;
    let mut rhs_buf = Buffer::from_vec(rhs, BufferUsage::Storage);
    rhs_buf.ensure_device(&device)?;
    rhs_buf.upload(&queue)?;

    let lower: Vec<Real> = damage_old.iter().map(|&d| d.clamp(0.0, 1.0)).collect();
    let mut low_buf = Buffer::from_vec(lower.clone(), BufferUsage::Storage);
    low_buf.ensure_device(&device)?;
    low_buf.upload(&queue)?;

    // Ping-pong damage buffers, seeded with the (clamped) initial value.
    let mut da = Buffer::from_vec(lower.clone(), BufferUsage::StorageReadWrite);
    da.ensure_device(&device)?;
    da.upload(&queue)?;
    let mut db = Buffer::from_vec(lower, BufferUsage::StorageReadWrite);
    db.ensure_device(&device)?;
    db.upload(&queue)?;

    let kernel = JacobiKernel::new(&device);
    let bg_mat =
        kernel.matrix_bind_group(&device, &row_buf, &col_buf, &val_buf, &rhs_buf, &low_buf)?;
    let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::pf uniforms"),
        size: core::mem::size_of::<Uniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(
        &uniform_buf,
        0,
        bytemuck::bytes_of(&Uniforms {
            n: [n as u32, 0, 0, 0],
        }),
    );
    let bg_uniform = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("molla::pf uniform bg"),
        layout: &kernel.layout_uniform,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buf.as_entire_binding(),
        }],
    });

    let da_dev = da.device_buffer().unwrap().clone();
    let db_dev = db.device_buffer().unwrap().clone();
    let bg_ab = kernel.pingpong_bind_group(&device, &da_dev, &db_dev)?; // in = da, out = db
    let bg_ba = kernel.pingpong_bind_group(&device, &db_dev, &da_dev)?; // in = db, out = da

    let groups = (n as u32).div_ceil(WORKGROUP_SIZE).max(1);
    for sweep in 0..sweeps {
        let bg_io = if sweep % 2 == 0 { &bg_ab } else { &bg_ba };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::pf encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::pf pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernel.pipeline);
            pass.set_bind_group(0, &bg_mat, &[]);
            pass.set_bind_group(1, bg_io, &[]);
            pass.set_bind_group(2, &bg_uniform, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    // After `sweeps` sweeps the newest values live in db (odd) or da (even).
    let result = if sweeps % 2 == 1 { &mut db } else { &mut da };
    result.mark_device_dirty();
    result.download(&device, &queue)?;
    Ok(result.host()?.to_vec())
}

struct JacobiKernel {
    pipeline: wgpu::ComputePipeline,
    layout_mat: wgpu::BindGroupLayout,
    layout_io: wgpu::BindGroupLayout,
    layout_uniform: wgpu::BindGroupLayout,
}

impl JacobiKernel {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::phase_field_jacobi"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/phase_field_jacobi.wgsl").into(),
            ),
        });
        let layout_mat = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::pf mat"),
            entries: &storage_entries(5, true),
        });
        let mut io_entries = storage_entries(1, true);
        io_entries.extend(storage_entries_from(1, 1, false));
        let layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::pf io"),
            entries: &io_entries,
        });
        let layout_uniform = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::pf uniform"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::pf layout"),
            bind_group_layouts: &[Some(&layout_mat), Some(&layout_io), Some(&layout_uniform)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::pf pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("jacobi_sweep"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            pipeline,
            layout_mat,
            layout_io,
            layout_uniform,
        }
    }

    fn matrix_bind_group(
        &self,
        device: &wgpu::Device,
        row: &wgpu::Buffer,
        col: &wgpu::Buffer,
        val: &Buffer<Real>,
        rhs: &Buffer<Real>,
        low: &Buffer<Real>,
    ) -> Result<wgpu::BindGroup> {
        let val = val.device_buffer().ok_or_else(dev_err)?;
        let rhs = rhs.device_buffer().ok_or_else(dev_err)?;
        let low = low.device_buffer().ok_or_else(dev_err)?;
        Ok(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::pf mat bg"),
            layout: &self.layout_mat,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: row.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: col.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: val.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: rhs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: low.as_entire_binding(),
                },
            ],
        }))
    }

    fn pingpong_bind_group(
        &self,
        device: &wgpu::Device,
        d_in: &wgpu::Buffer,
        d_out: &wgpu::Buffer,
    ) -> Result<wgpu::BindGroup> {
        Ok(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::pf io bg"),
            layout: &self.layout_io,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: d_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: d_out.as_entire_binding(),
                },
            ],
        }))
    }
}

fn dev_err() -> Error {
    Error::Gpu("phase_field_gpu: no device mirror".into())
}

fn make_storage(device: &wgpu::Device, label: &str, data: &[u8]) -> wgpu::Buffer {
    let size = (data.len().max(16)) as u64;
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn storage_entries(count: u32, read_only: bool) -> Vec<wgpu::BindGroupLayoutEntry> {
    storage_entries_from(0, count, read_only)
}

fn storage_entries_from(
    start: u32,
    count: u32,
    read_only: bool,
) -> Vec<wgpu::BindGroupLayoutEntry> {
    (0..count)
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: start + i,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect()
}
