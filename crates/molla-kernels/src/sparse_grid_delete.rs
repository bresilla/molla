//! Stage 24e.2 — GPU-side mutable delete + leaf reclamation.
//!
//! Wraps `shaders/sparse_grid_delete.wgsl`. One
//! [`SparseGridDeleteKernel`] owns the WGSL pipelines (one per value
//! type — `f32`, `vec3<f32>`, `f16`, mirroring
//! [`super::SparseGridInsertKernel`]). Per-dispatch the caller
//! supplies a list of integer voxel coords to mark inactive and the
//! same persistent [`super::SparseGridStorage`] used for inserts.
//!
//! ## Lifecycle
//!
//! Typical per-frame sequence:
//!
//! 1. `mark_inactive(coords)` — clears one bit per coord in the
//!    matching leaf mask. If a leaf goes fully empty it lands on the
//!    `pending_reclaim` LIFO.
//! 2. `reclaim()` — drains `pending_reclaim` into `freelist`, ready
//!    for the next round of `set_value` to recycle the slots.
//!
//! The two passes are split so that concurrent inserters don't race
//! the deleter on the same leaf within a single dispatch; the host
//! is expected to insert a `queue.submit` boundary between them.
//!
//! ## Caveats
//!
//! - **No internal-node reclamation in v1.** When all leaves under
//!   an internal node are reclaimed, the internal node + its mask
//!   bit + the `internal_children[slot]` pointer all stay live. A
//!   subsequent insert at *exactly the same logical coord* hits the
//!   stale parent pointer and silently re-uses the leaf in place
//!   (the leaf's mask is zero, so writes are valid). A subsequent
//!   insert that the freelist hands the same leaf to a *different*
//!   coord will leave the original parent pointer dangling — the
//!   leaf is now reachable from two places. Reads at the original
//!   coord still return the background because the leaf-local mask
//!   bit at the original coord's offset is unset. Reads at the new
//!   coord see the new value. Cascade-delete + parent fixup is
//!   deferred to Stage 24e.3.
//! - **Atomic clear via `atomicAnd<u32>`.** WGSL guarantees this
//!   primitive on storage buffers. If a future wgpu backend lacks
//!   it, swap the `atomicAnd(&w, ~bit)` call in
//!   `mark_inactive_voxel` for a CAS loop reading `atomicLoad(&w)`,
//!   computing `w & ~bit`, and looping on
//!   `atomicCompareExchangeWeak(&w, w, new)` until it succeeds.
//! - **Pending-list overflow** (more reclaim candidates than the
//!   storage's `leaf_capacity`) silently drops the excess. The
//!   leaves become "stranded" — mask is zero, no recycling — but
//!   topology stays consistent. Sized to the leaf pool, this should
//!   only happen if the host calls `mark_inactive` with more coords
//!   than the pool itself can hold, which is already an error.

use crate::sparse_grid_insert::{COUNTER_SLOTS, SparseGridStorage, Value};
use bytemuck::{Pod, Zeroable};
use molla_core::Result;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct DeleteUniforms {
    count: u32,
    root_table_capacity: u32,
    internal_capacity: u32,
    leaf_capacity: u32,
}

const WORKGROUP_SIZE: u32 = 64;

/// Compiled WGSL pipelines for the delete kernel. Three entry
/// points: `mark_inactive_voxel`, `reclaim_empty_leaves`,
/// `reset_pending_counter`.
///
/// `V` parameterises over the same value type as the matching
/// [`super::SparseGridStorage`] / [`super::SparseGridInsertKernel`]
/// — needed because the bind-group layout matches those kernels'
/// shape, but the delete kernel itself never touches `leaf_values`,
/// so the type only matters for the storage compatibility.
pub struct SparseGridDeleteKernel<V: Value> {
    pipeline_mark_inactive: wgpu::ComputePipeline,
    pipeline_reclaim: wgpu::ComputePipeline,
    pipeline_reset_pending: wgpu::ComputePipeline,
    layout_storage: wgpu::BindGroupLayout,
    layout_io: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
    _phantom: core::marker::PhantomData<V>,
}

impl<V: Value> SparseGridDeleteKernel<V> {
    pub fn new(device: &wgpu::Device) -> Self {
        let wgsl = include_str!("../shaders/sparse_grid_delete.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::sparse_grid_delete"),
            source: wgpu::ShaderSource::Wgsl(wgsl.into()),
        });

        // 6 storage entries: root_table, internal_children,
        // internal_masks, leaf_masks, freelist, pending_reclaim.
        // (No leaf_values — delete never reads/writes them.)
        let layout_storage = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::sparse_grid_delete storage"),
            entries: &(0..6)
                .map(|i| wgpu::BindGroupLayoutEntry {
                    binding: i,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::sparse_grid_delete io"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::sparse_grid_delete uniforms"),
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
            label: Some("molla::sparse_grid_delete layout"),
            bind_group_layouts: &[&layout_storage, &layout_io, &layout_uniforms],
            push_constant_ranges: &[],
        });

        let make_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        Self {
            pipeline_mark_inactive: make_pipeline("mark_inactive_voxel"),
            pipeline_reclaim: make_pipeline("reclaim_empty_leaves"),
            pipeline_reset_pending: make_pipeline("reset_pending_counter"),
            layout_storage,
            layout_io,
            layout_uniforms,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Mark every coord in `coords` inactive. Returns the post-
    /// dispatch counter snapshot (`[next_internal, next_leaf,
    /// error_flags, freelist_count, pending_reclaim_count, _]`).
    ///
    /// The freelist count won't have changed yet — the reclaim pass
    /// is a separate dispatch. Use [`Self::reclaim`] to drain
    /// `pending_reclaim` into `freelist`.
    pub fn mark_inactive(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        storage: &SparseGridStorage<V>,
        coords: &[[i32; 3]],
    ) -> Result<[u32; COUNTER_SLOTS as usize]> {
        if coords.is_empty() {
            return storage.read_all_counters(device, queue);
        }

        // Pad coords to vec4<i32>.
        let mut padded: Vec<i32> = Vec::with_capacity(coords.len() * 4);
        for c in coords {
            padded.push(c[0]);
            padded.push(c[1]);
            padded.push(c[2]);
            padded.push(0);
        }
        let buf_coords = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("molla::sparse_grid_delete query_coords"),
            contents: bytemuck::cast_slice(&padded),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let uniforms = DeleteUniforms {
            count: coords.len() as u32,
            root_table_capacity: storage.root_table_capacity,
            internal_capacity: storage.internal_capacity,
            leaf_capacity: storage.leaf_capacity,
        };
        let buf_uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("molla::sparse_grid_delete uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bg_storage = self.bg_storage(device, storage, "mark_inactive");
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_grid_delete mark_inactive io bg"),
            layout: &self.layout_io,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage.counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_coords.as_entire_binding(),
                },
            ],
        });
        let bg_u = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_grid_delete mark_inactive uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_uniforms.as_entire_binding(),
            }],
        });

        let groups = (coords.len() as u32).div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::sparse_grid_delete mark_inactive encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::sparse_grid_delete mark_inactive pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_mark_inactive);
            pass.set_bind_group(0, &bg_storage, &[]);
            pass.set_bind_group(1, &bg_io, &[]);
            pass.set_bind_group(2, &bg_u, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        storage.read_all_counters(device, queue)
    }

    /// Drain `pending_reclaim` into `freelist`, then zero out the
    /// pending counter. After this call returns, `freelist_count`
    /// reflects the recycled-leaf depth ready for the next insert.
    pub fn reclaim(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        storage: &SparseGridStorage<V>,
    ) -> Result<[u32; COUNTER_SLOTS as usize]> {
        // Read the pending count to size the dispatch. (Cheap; one
        // 4-byte readback.) If zero we still emit the reset kernel
        // to keep counter hygiene consistent across calls.
        let counters = storage.read_all_counters(device, queue)?;
        let pending = counters[4];

        // Stub coords buffer — bind_group_layout requires the same
        // binding even though `reclaim_empty_leaves` doesn't read
        // from it. wgpu's "min binding size" defaults to 4.
        let stub_coords = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::sparse_grid_delete reclaim stub_coords"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniforms = DeleteUniforms {
            count: pending,
            root_table_capacity: storage.root_table_capacity,
            internal_capacity: storage.internal_capacity,
            leaf_capacity: storage.leaf_capacity,
        };
        let buf_uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("molla::sparse_grid_delete reclaim uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bg_storage = self.bg_storage(device, storage, "reclaim");
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_grid_delete reclaim io bg"),
            layout: &self.layout_io,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage.counters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: stub_coords.as_entire_binding(),
                },
            ],
        });
        let bg_u = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::sparse_grid_delete reclaim uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_uniforms.as_entire_binding(),
            }],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::sparse_grid_delete reclaim encoder"),
        });
        if pending > 0 {
            let groups = pending.div_ceil(WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::sparse_grid_delete reclaim pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_reclaim);
            pass.set_bind_group(0, &bg_storage, &[]);
            pass.set_bind_group(1, &bg_io, &[]);
            pass.set_bind_group(2, &bg_u, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        // Always emit the reset — keeps the pending counter at zero
        // for the next round even if we had nothing to drain.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::sparse_grid_delete reset_pending pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_reset_pending);
            pass.set_bind_group(0, &bg_storage, &[]);
            pass.set_bind_group(1, &bg_io, &[]);
            pass.set_bind_group(2, &bg_u, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        storage.read_all_counters(device, queue)
    }

    /// Convenience helper: `mark_inactive` followed by `reclaim` in
    /// one call. Two `queue.submit`s under the hood — needed to
    /// guarantee the mark pass globally finishes before the drain
    /// reads the pending list.
    ///
    /// # Reclamation contract (INSPECTION.md F11)
    ///
    /// * **Leaf slots are recycled end-to-end**: emptied leaves flow
    ///   through `pending_reclaim` into the freelist, and the insert
    ///   kernel's `pop_freelist` consumes them before growing the
    ///   bump allocator — verified by
    ///   `sparse_grid_v4::leaf_slots_recycle_but_internal_nodes_leak_known_limit`
    ///   (a full region's re-population grows the leaf pool by ~0).
    /// * **Internal nodes are NOT folded**: abandoning a region leaks
    ///   its internal-node slots (there is no internal freelist and no
    ///   empty-subtree coarsening). The leak is bounded by
    ///   `internal_capacity`; `SparseGridStorage::reset` reclaims
    ///   everything. Workloads that continuously migrate across
    ///   root-key space will exhaust the internal pool and should
    ///   reset periodically until folding lands.
    pub fn delete(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        storage: &SparseGridStorage<V>,
        coords: &[[i32; 3]],
    ) -> Result<[u32; COUNTER_SLOTS as usize]> {
        let _ = self.mark_inactive(device, queue, storage, coords)?;
        self.reclaim(device, queue, storage)
    }

    fn bg_storage(
        &self,
        device: &wgpu::Device,
        storage: &SparseGridStorage<V>,
        label: &str,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("molla::sparse_grid_delete {label} storage bg")),
            layout: &self.layout_storage,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: storage.root_table.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: storage.internal_children.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: storage.internal_masks.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: storage.leaf_masks.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: storage.freelist.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: storage.pending_reclaim.as_entire_binding(),
                },
            ],
        })
    }
}
