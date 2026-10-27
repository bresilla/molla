//! GPU device-wide u32 scan (exclusive + inclusive) over a single
//! storage buffer of arbitrary length.
//!
//! Stage 38e — replaces the single-workgroup Hillis-Steele scan in
//! `shaders/tile_stream_compaction.wgsl` for the cases that need
//! n >> workgroup_size (sparse-grid insert, hash-grid buckets, contact
//! pair compaction, SpGEMM).
//!
//! ### Algorithm
//!
//! Two-pass scan-then-fixup, matching Warp's `scan.cu` / NVIDIA Harris
//! 2007:
//!
//!   1. `scan_local` — per-workgroup exclusive scan into `output`, with
//!      the per-WG total emitted into `group_sums[wg]`.
//!   2. `scan_groups` — single-workgroup exclusive scan over
//!      `group_sums`. Tiles internally so a single dispatch covers up
//!      to `WG_SIZE^2 = 65 536` groups → 16 M elements before a
//!      hierarchical fallback is needed.
//!   3. `scan_fixup` — adds `group_sums[wg]` to every element; in the
//!      inclusive variant also adds `input[i]`.

use std::sync::Arc;

use molla_core::{Error, Result};

const WG_SIZE: u32 = 256;
/// Maximum N that pass 2's single-WG scan can cover in one shot
/// (`WG_SIZE` groups per tile × `WG_SIZE` tiles).
const MAX_GROUPS: u32 = WG_SIZE * WG_SIZE; // 65 536 groups → 16 M elements

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    n: u32,
    inclusive: u32,
    num_groups: u32,
    _pad: u32,
}

/// Reusable GPU u32-scan dispatcher.
///
/// **Concurrency:** the internal `group_sums` + `uniforms` buffers
/// are shared across calls on a single instance. Issuing a second
/// scan before the first's GPU submission has fully drained will
/// race on those buffers — the second scan's pass-1 may read the
/// first scan's stale `group_sums`. For pipelined usage (multiple
/// scans in flight) construct one `ScanKernel` per logical stream,
/// or wait for `device.poll(...)` between calls. Single-threaded
/// "submit, poll, submit" usage is safe.
pub struct ScanKernel {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    pipeline_local: wgpu::ComputePipeline,
    pipeline_groups: wgpu::ComputePipeline,
    pipeline_fixup: wgpu::ComputePipeline,

    layout_data: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,

    /// Per-WG totals; sized to fit `MAX_GROUPS` worst case (256 KiB).
    /// Allocated once at construction.
    group_sums: wgpu::Buffer,

    /// Two uniform buffers — one for the exclusive variant, one for
    /// inclusive — so a back-to-back inclusive + exclusive in the same
    /// frame don't stomp each other before the GPU has caught up.
    uniforms: wgpu::Buffer,
}

impl ScanKernel {
    /// Build pipelines + scratch buffers. Allocates a single
    /// `group_sums` buffer sized for the maximum supported `n`
    /// (`WG_SIZE^2 * WG_SIZE = 16 M elements`).
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::scan"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/scan.wgsl").into()),
        });

        let layout_data = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::scan data"),
            entries: &[ro_storage(0), rw_storage(1), rw_storage(2)],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::scan uniforms"),
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
            label: Some("molla::scan layout"),
            bind_group_layouts: &[Some(&layout_data), Some(&layout_uniforms)],
            immediate_size: 0,
        });

        let pipeline_local = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::scan::scan_local"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("scan_local"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_groups = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::scan::scan_groups"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("scan_groups"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_fixup = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::scan::scan_fixup"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("scan_fixup"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let group_sums = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::scan group_sums"),
                size: (MAX_GROUPS as u64) * core::mem::size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );

        let uniforms = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::scan uniforms"),
                size: core::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );

        Self {
            device,
            queue,
            pipeline_local,
            pipeline_groups,
            pipeline_fixup,
            layout_data,
            layout_uniforms,
            group_sums,
            uniforms,
        }
    }

    /// Borrow of the internal `group_sums` buffer — used by
    /// `RadixSortKernel64` so we don't allocate twice.
    #[allow(dead_code)]
    pub(crate) fn group_sums_buf(&self) -> &wgpu::Buffer {
        &self.group_sums
    }

    /// Exclusive scan: `out[i] = sum(in[0..i])`. `output` length must
    /// equal `input` length.
    pub fn exclusive_scan_u32(
        &self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        n: u32,
    ) -> Result<()> {
        self.scan_internal(input, output, n, /* inclusive = */ false)
    }

    /// Inclusive scan: `out[i] = sum(in[0..=i])`.
    pub fn inclusive_scan_u32(
        &self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        n: u32,
    ) -> Result<()> {
        self.scan_internal(input, output, n, /* inclusive = */ true)
    }

    /// Convenience: returns the total sum (last element of inclusive
    /// scan) as a host `u64`. Adds a small readback dispatch — use
    /// sparingly.
    pub fn total_sum_u32(&self, input: &wgpu::Buffer, n: u32) -> Result<u64> {
        if n == 0 {
            return Ok(0);
        }
        // Run an inclusive scan into a transient buffer, then read its
        // last u32 back.
        let bytes = (n as u64) * core::mem::size_of::<u32>() as u64;
        let scratch = molla_core::gpu_create_buffer!(
            self.device,
            &wgpu::BufferDescriptor {
                label: Some("molla::scan::total_sum scratch"),
                size: bytes.max(16),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        self.inclusive_scan_u32(input, &scratch, n)?;

        let staging = molla_core::gpu_create_buffer!(
            self.device,
            &wgpu::BufferDescriptor {
                label: Some("molla::scan::total_sum staging"),
                size: core::mem::size_of::<u32>() as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::scan::total_sum readback"),
            });
        let last_off = ((n as u64) - 1) * core::mem::size_of::<u32>() as u64;
        encoder.copy_buffer_to_buffer(
            &scratch,
            last_off,
            &staging,
            0,
            core::mem::size_of::<u32>() as u64,
        );
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("scan::total_sum poll: {e:?}")))?;
        rx.recv()
            .map_err(|e| Error::Gpu(format!("scan::total_sum recv: {e}")))?
            .map_err(|e| Error::Gpu(format!("scan::total_sum map: {e:?}")))?;
        let mapped = molla_core::gpu_readback!(slice);
        let last = u32::from_le_bytes([mapped[0], mapped[1], mapped[2], mapped[3]]);
        drop(mapped);
        staging.unmap();
        Ok(last as u64)
    }

    fn scan_internal(
        &self,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
        n: u32,
        inclusive: bool,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let num_groups = n.div_ceil(WG_SIZE);
        if num_groups > MAX_GROUPS {
            return Err(Error::Gpu(format!(
                "scan: n={n} exceeds max-groups (max ~{} elements)",
                MAX_GROUPS * WG_SIZE
            )));
        }

        let u = Uniforms {
            n,
            inclusive: u32::from(inclusive),
            num_groups,
            _pad: 0,
        };
        molla_core::gpu_write_buffer!(self.queue, &self.uniforms, 0, bytemuck::bytes_of(&u));

        let bg_data = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::scan data bg"),
            layout: &self.layout_data,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.group_sums.as_entire_binding(),
                },
            ],
        });
        let bg_uniforms = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::scan uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.uniforms.as_entire_binding(),
            }],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::scan encoder"),
            });

        // Pass 1: per-WG exclusive scan + group totals.
        {
            let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::scan::scan_local"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.pipeline_local);
            p.set_bind_group(0, &bg_data, &[]);
            p.set_bind_group(1, &bg_uniforms, &[]);
            p.dispatch_workgroups(num_groups, 1, 1);
        }
        // Pass 2: scan over group_sums (single WG, internally tiled).
        {
            let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::scan::scan_groups"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.pipeline_groups);
            p.set_bind_group(0, &bg_data, &[]);
            p.set_bind_group(1, &bg_uniforms, &[]);
            p.dispatch_workgroups(1, 1, 1);
        }
        // Pass 3: fixup (and inclusive convert if requested).
        {
            let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::scan::scan_fixup"),
                timestamp_writes: None,
            });
            p.set_pipeline(&self.pipeline_fixup);
            p.set_bind_group(0, &bg_data, &[]);
            p.set_bind_group(1, &bg_uniforms, &[]);
            p.dispatch_workgroups(num_groups, 1, 1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }
}

fn ro_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn rw_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
