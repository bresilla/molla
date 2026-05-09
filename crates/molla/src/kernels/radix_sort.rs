//! GPU radix sort over `(key=u32, value=u32)` pairs.
//!
//! 4 passes of 8-bit radix → 32-bit keys. Each pass uses three
//! dispatches (`count_block_bins`, `scan_global_bins`, `scatter`) and
//! a double-buffer ping-pong: pass 0 reads from caller-supplied
//! `(keys, values)` and writes to a pair of internal scratch buffers;
//! pass 1 reads from scratch and writes back to the caller buffers;
//! and so on. After 4 passes the final sorted result lands back in
//! the caller's buffers.
//!
//! ### Algorithm choice — radix vs bitonic vs merge
//!
//! See the WGSL header in `shaders/radix_sort.wgsl` for the full
//! rationale. TL;DR: 4-pass 8-bit radix is the textbook GPU LBVH
//! choice (Karras 2012, OptiX, Embree, NVIDIA Warp / cuBQL all do
//! this). Bitonic is O(n log² n) which loses badly past n ~ 10k;
//! merge sort hits memory bandwidth before radix does.
//!
//! Stage 19.1 — replaces the CPU-sort stop-gap that Stage 19 shipped.
//!
//! ### Stable sort
//!
//! The scatter kernel processes each block's elements in input-index
//! order from a single thread, so the i-th occurrence of digit `d` in
//! block `b` always lands at offset `block_bins[d, b] + i`. Across
//! blocks, the global scan visits (digit, block) in row-major order
//! so earlier blocks always get lower output offsets. Stability is
//! load-bearing for LBVH: ties on Morton codes need to break
//! deterministically so the WGSL `delta` function in `lbvh.wgsl`
//! sees the same ordering the build pass expects.

use std::sync::Arc;

use crate::core::{Error, Result};

const BLOCK_SIZE: u32 = 256;
const NUM_BINS: u32 = 256;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    n: u32,
    shift: u32,
    num_blocks: u32,
    _pad: u32,
}

/// Reusable GPU radix-sort dispatcher. Holds the three compute
/// pipelines + two layouts and grows scratch buffers as needed.
pub struct RadixSortKernel {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    pipeline_count: wgpu::ComputePipeline,
    pipeline_scan: wgpu::ComputePipeline,
    pipeline_scatter: wgpu::ComputePipeline,

    layout_data: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,

    /// Scratch buffers for ping-pong + per-pass uniforms + block_bins.
    scratch_keys: wgpu::Buffer,
    scratch_values: wgpu::Buffer,
    block_bins: wgpu::Buffer,
    /// Four uniform buffers — one per pass — so encode-then-submit
    /// in one go without `write_buffer` between passes.
    uniforms: [wgpu::Buffer; 4],

    /// Capacity (in elements) of `scratch_keys` / `scratch_values`.
    capacity: usize,
    /// Capacity (in u32s) of `block_bins`.
    bins_capacity: usize,
}

impl RadixSortKernel {
    /// Build pipelines + initial scratch buffers sized for
    /// `initial_capacity` keys.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        initial_capacity: usize,
    ) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::radix_sort"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/radix_sort.wgsl").into(),
            ),
        });

        let layout_data = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::radix_sort data"),
            entries: &[
                ro_storage(0),
                ro_storage(1),
                rw_storage(2),
                rw_storage(3),
                rw_storage(4),
            ],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::radix_sort uniforms"),
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
            label: Some("molla::radix_sort layout"),
            bind_group_layouts: &[&layout_data, &layout_uniforms],
            push_constant_ranges: &[],
        });

        let pipeline_count = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::radix_sort::count_block_bins"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("count_block_bins"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_scan = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::radix_sort::scan_global_bins"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("scan_global_bins"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_scatter = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::radix_sort::scatter"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("scatter"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let cap = initial_capacity.max(BLOCK_SIZE as usize);
        let scratch_keys = alloc_storage(
            &device,
            "molla::radix_sort scratch_keys",
            (cap * core::mem::size_of::<u32>()) as u64,
        );
        let scratch_values = alloc_storage(
            &device,
            "molla::radix_sort scratch_values",
            (cap * core::mem::size_of::<u32>()) as u64,
        );
        let bins_cap = bins_capacity_for(cap);
        let block_bins = alloc_storage(
            &device,
            "molla::radix_sort block_bins",
            (bins_cap * core::mem::size_of::<u32>()) as u64,
        );

        let uniforms = std::array::from_fn(|i| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(match i {
                    0 => "molla::radix_sort uniforms[0]",
                    1 => "molla::radix_sort uniforms[1]",
                    2 => "molla::radix_sort uniforms[2]",
                    _ => "molla::radix_sort uniforms[3]",
                }),
                size: core::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        Ok(Self {
            device,
            queue,
            pipeline_count,
            pipeline_scan,
            pipeline_scatter,
            layout_data,
            layout_uniforms,
            scratch_keys,
            scratch_values,
            block_bins,
            uniforms,
            capacity: cap,
            bins_capacity: bins_cap,
        })
    }

    /// Element capacity of the scratch ping-pong buffers.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn ensure_capacity(&mut self, n: usize) {
        if n <= self.capacity {
            return;
        }
        let cap = (n * 2).max(self.capacity * 2).max(BLOCK_SIZE as usize);
        self.scratch_keys = alloc_storage(
            &self.device,
            "molla::radix_sort scratch_keys",
            (cap * core::mem::size_of::<u32>()) as u64,
        );
        self.scratch_values = alloc_storage(
            &self.device,
            "molla::radix_sort scratch_values",
            (cap * core::mem::size_of::<u32>()) as u64,
        );
        let bins_cap = bins_capacity_for(cap);
        if bins_cap > self.bins_capacity {
            self.block_bins = alloc_storage(
                &self.device,
                "molla::radix_sort block_bins",
                (bins_cap * core::mem::size_of::<u32>()) as u64,
            );
            self.bins_capacity = bins_cap;
        }
        self.capacity = cap;
    }

    /// Sort `len` elements of `(keys, values)` in place, ascending by
    /// key. `keys` and `values` must each be at least
    /// `len * size_of::<u32>()` bytes and have `STORAGE | COPY_SRC |
    /// COPY_DST` usage. Stable.
    pub fn dispatch(
        &mut self,
        keys: &wgpu::Buffer,
        values: &wgpu::Buffer,
        len: usize,
    ) -> Result<()> {
        if len < 2 {
            return Ok(());
        }
        self.ensure_capacity(len);

        let n = len as u32;
        let num_blocks = n.div_ceil(BLOCK_SIZE);

        // Stage uniform buffers — one per pass (shifts 0, 8, 16, 24).
        for pass in 0..4 {
            let u = Uniforms {
                n,
                shift: (pass * 8) as u32,
                num_blocks,
                _pad: 0,
            };
            self.queue
                .write_buffer(&self.uniforms[pass], 0, bytemuck::bytes_of(&u));
        }

        // Encode all 4 passes in one command encoder.
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::radix_sort encoder"),
            });

        // Pass alternates the (in, out) pair. We always read from
        // `caller`-supplied buffers on pass 0 and end with the result
        // back in caller buffers after 4 passes.
        // pass 0: caller → scratch
        // pass 1: scratch → caller
        // pass 2: caller → scratch
        // pass 3: scratch → caller
        for pass in 0..4 {
            let (in_keys, in_values, out_keys, out_values) = if pass % 2 == 0 {
                (keys, values, &self.scratch_keys, &self.scratch_values)
            } else {
                (&self.scratch_keys, &self.scratch_values, keys, values)
            };
            let bg_data = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::radix_sort data bg"),
                layout: &self.layout_data,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: in_keys.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: in_values.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_keys.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: out_values.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.block_bins.as_entire_binding(),
                    },
                ],
            });
            let bg_uniforms = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::radix_sort uniforms bg"),
                layout: &self.layout_uniforms,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniforms[pass].as_entire_binding(),
                }],
            });

            // Count.
            {
                let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("molla::radix_sort::count"),
                    timestamp_writes: None,
                });
                p.set_pipeline(&self.pipeline_count);
                p.set_bind_group(0, &bg_data, &[]);
                p.set_bind_group(1, &bg_uniforms, &[]);
                p.dispatch_workgroups(num_blocks, 1, 1);
            }
            // Scan.
            {
                let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("molla::radix_sort::scan"),
                    timestamp_writes: None,
                });
                p.set_pipeline(&self.pipeline_scan);
                p.set_bind_group(0, &bg_data, &[]);
                p.set_bind_group(1, &bg_uniforms, &[]);
                p.dispatch_workgroups(1, 1, 1);
            }
            // Scatter.
            {
                let mut p = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("molla::radix_sort::scatter"),
                    timestamp_writes: None,
                });
                p.set_pipeline(&self.pipeline_scatter);
                p.set_bind_group(0, &bg_data, &[]);
                p.set_bind_group(1, &bg_uniforms, &[]);
                p.dispatch_workgroups(num_blocks, 1, 1);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }
}

/// Convenience free function — builds a transient kernel and runs one
/// sort. Prefer creating a [`RadixSortKernel`] and reusing it across
/// frames so the pipelines + scratch buffers don't get re-allocated.
pub fn radix_sort_pairs(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    keys: &wgpu::Buffer,
    values: &wgpu::Buffer,
    len: usize,
) -> Result<()> {
    let mut k = RadixSortKernel::new(device, queue, len)?;
    k.dispatch(keys, values, len)
}

fn bins_capacity_for(n_keys: usize) -> usize {
    // ceil(n / BLOCK) blocks × NUM_BINS bins each. Always at least
    // NUM_BINS (1 block).
    let blocks = n_keys.div_ceil(BLOCK_SIZE as usize).max(1);
    blocks * NUM_BINS as usize
}

fn alloc_storage(device: &wgpu::Device, label: &str, size: u64) -> wgpu::Buffer {
    let size = size.max(16);
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
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

// Quiet unused-import warning when only Error is referenced indirectly.
#[allow(dead_code)]
fn _touch_error(_: Error) {}
