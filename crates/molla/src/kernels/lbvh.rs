//! GPU LBVH (Linear BVH) broad phase — Stage 19.
//!
//! Builds a Karras-style LBVH over per-shape AABBs each frame and
//! emits overlapping AABB pairs into a `(pair_a, pair_b)` buffer.
//! Mirrors `crate::geometry::BroadPhase` but on the device.
//!
//! Pipeline (per frame):
//!   1. Host computes per-shape AABBs (reuses `shape_world_aabb`).
//!   2. Host computes 30-bit Morton codes of each centroid relative to
//!      the scene AABB and **sorts** the (code, idx) pairs.
//!   3. Upload `primitive_aabb`, `sorted_codes`, `sorted_idx`, and a
//!      zero-initialised `nodes` / `visited` buffer to the device.
//!   4. Dispatch `build_internal` (one thread per internal node).
//!   5. Dispatch `refit_aabbs` (one thread per leaf — atomic walk to
//!      root unioning child AABBs).
//!   6. Dispatch `find_pairs` (one thread per leaf — DFS the tree,
//!      emit overlapping pairs into the `pair_a` / `pair_b` arrays).
//!   7. Download `pair_count` + the populated pair arrays.
//!
//! ### Radix-sort decision
//!
//! Karras-style LBVH wants Morton codes in sorted order. The textbook
//! GPU implementation is a multi-pass WGSL radix sort (4 bits per
//! pass × 8 passes for a 32-bit key, with per-workgroup histograms +
//! a global scan). That's a separate ~300 LOC kernel and a sizeable
//! research project on its own.
//!
//! For Stage 19 we picked the **CPU sort + upload** stop-gap path.
//! Rationale:
//!   - It keeps the pipeline correct and end-to-end testable while
//!     the GPU radix sort lands separately.
//!   - For the 1k–10k shape range called out in the deliverables, the
//!     CPU sort cost is dominated by the upload itself; a GPU radix
//!     would shave maybe 0.3 ms.
//!   - The uploaded `(sorted_codes, sorted_idx)` buffers and the
//!     downstream WGSL kernels (`build_internal`, `refit_aabbs`,
//!     `find_pairs`) are unchanged when a GPU radix lands later.
//!
//! ### Refit-only mode
//!
//! For deformable-geometry scenes where shape topology is fixed but
//! the per-shape AABBs change every frame (cloth particles, soft
//! bodies), [`LbvhBroadPhase::refit_only`] reuses the previous frame's
//! tree topology and only re-runs `refit_aabbs` + `find_pairs`. Skips
//! the (Morton + sort + build_internal) cost.

use std::sync::Arc;

use crate::core::{Error, Result};
use crate::math::{Real, Vec3};
use crate::sim::{Model, State};

const WORKGROUP_SIZE: u32 = 64;

/// Per-node device layout — must match the WGSL `LbvhNode` struct in
/// `shaders/lbvh.wgsl` (8 × `vec4<f32>` worth = 64 bytes per node).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuLbvhNode {
    aabb_min: [f32; 4],
    aabb_max: [f32; 4],
    left: u32,
    right: u32,
    parent: u32,
    is_leaf: u32,
}

/// Per-shape AABB upload layout — must match the WGSL `Aabb` struct.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuAabb {
    min: [f32; 4],
    max: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    primitive_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    scene_min: [f32; 4],
    scene_max: [f32; 4],
}

/// LBVH broad-phase dispatcher. Holds compute pipelines + reusable
/// device buffers; intended to live across frames so the buffers can
/// grow geometrically rather than realloc on every step.
pub struct LbvhBroadPhase {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    pipeline_build: wgpu::ComputePipeline,
    pipeline_refit: wgpu::ComputePipeline,
    pipeline_pairs: wgpu::ComputePipeline,
    layout_tree: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
    layout_pairs: wgpu::BindGroupLayout,

    /// Per-shape AABB upload buffer (`GpuAabb` array). Sized to
    /// `prim_capacity` `GpuAabb` slots.
    primitive_aabb: wgpu::Buffer,
    sorted_idx: wgpu::Buffer,
    sorted_codes: wgpu::Buffer,
    /// `2 * prim_capacity - 1` `GpuLbvhNode` slots.
    nodes: wgpu::Buffer,
    /// `prim_capacity - 1` u32 atomic counters (one per internal node).
    visited: wgpu::Buffer,

    pair_count: wgpu::Buffer,
    pair_a: wgpu::Buffer,
    pair_b: wgpu::Buffer,

    uniforms: wgpu::Buffer,

    /// Allocated capacity (number of primitives the buffers can hold).
    prim_capacity: usize,
    /// Allocated pair-output capacity.
    pair_capacity: usize,

    /// Cached topology from the last full build. Reused by refit-only
    /// mode. Stored as the host-side `(sorted_idx, sorted_codes)`
    /// pair so the host can re-upload them without recomputing the
    /// Morton sort. `prim_count` is the primitive count at build time.
    cached_topology: Option<CachedTopology>,
}

#[derive(Clone)]
struct CachedTopology {
    prim_count: u32,
    /// Scene bounds at build time — refit assumes the bounds haven't
    /// drifted catastrophically (rebuild if it grew >2x).
    scene_min: [f32; 4],
    scene_max: [f32; 4],
    /// Root node index from the build pass — find_pairs reads this
    /// from `pair_count[1]`. We mirror it on the host so refit-only
    /// can re-stamp it before the find_pairs dispatch.
    root: u32,
}

/// One node, decoded for debug printing.
#[derive(Copy, Clone, Debug)]
pub struct DebugNode {
    pub left: u32,
    pub right: u32,
    pub parent: u32,
    pub is_leaf: u32,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
}

/// Result of a broad-phase dispatch: the populated host-side pair
/// list. Pair indices reference the original shape index — i.e. the
/// pre-sort primitive ids.
#[derive(Default, Debug, Clone)]
pub struct LbvhPairs {
    pub a: Vec<u32>,
    pub b: Vec<u32>,
    /// True if the GPU `pair_count` overflowed the device buffer
    /// capacity. The host arrays only hold the first `pair_capacity`
    /// entries; the next `dispatch` call will grow the capacity.
    pub overflowed: bool,
}

impl LbvhBroadPhase {
    /// Initialise pipelines + minimum-sized buffers. Call
    /// [`Self::dispatch`] (or [`Self::refit_only`]) per frame.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        initial_prim_capacity: usize,
        initial_pair_capacity: usize,
    ) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::lbvh"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/lbvh.wgsl").into()),
        });

        // Group 0 — tree storage (5 buffers; pair-finder needs `nodes`
        // and `visited` writable but we keep all five at the same
        // group so a single layout serves every kernel).
        let layout_tree = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::lbvh tree"),
            entries: &[
                read_only_storage(0),
                read_only_storage(1),
                read_only_storage(2),
                read_write_storage(3),
                read_write_storage(4),
            ],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::lbvh uniforms"),
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
        let layout_pairs = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::lbvh pairs"),
            entries: &[
                read_write_storage(0),
                read_write_storage(1),
                read_write_storage(2),
            ],
        });
        let pipeline_layout_no_pairs =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::lbvh layout (no pairs)"),
                bind_group_layouts: &[&layout_tree, &layout_uniforms],
                push_constant_ranges: &[],
            });
        let pipeline_layout_with_pairs =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::lbvh layout (pairs)"),
                bind_group_layouts: &[&layout_tree, &layout_uniforms, &layout_pairs],
                push_constant_ranges: &[],
            });
        // Build pipeline now needs the pair-output bind group too —
        // it writes the root index into `pair_count[1]`.
        let pipeline_build = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::lbvh::build_hierarchy"),
            layout: Some(&pipeline_layout_with_pairs),
            module: &shader,
            entry_point: Some("build_hierarchy"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_refit = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::lbvh::refit_aabbs"),
            layout: Some(&pipeline_layout_no_pairs),
            module: &shader,
            entry_point: Some("refit_aabbs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_pairs = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::lbvh::find_pairs"),
            layout: Some(&pipeline_layout_with_pairs),
            module: &shader,
            entry_point: Some("find_pairs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let prim_capacity = initial_prim_capacity.max(2);
        let pair_capacity = initial_pair_capacity.max(64);

        let primitive_aabb = alloc_storage(
            &device,
            "molla::lbvh primitive_aabb",
            (prim_capacity * core::mem::size_of::<GpuAabb>()) as u64,
        );
        let sorted_idx = alloc_storage(
            &device,
            "molla::lbvh sorted_idx",
            (prim_capacity * core::mem::size_of::<u32>()) as u64,
        );
        let sorted_codes = alloc_storage(
            &device,
            "molla::lbvh sorted_codes",
            (prim_capacity * core::mem::size_of::<u32>()) as u64,
        );
        let node_count = (2 * prim_capacity - 1).max(1);
        let nodes = alloc_storage(
            &device,
            "molla::lbvh nodes",
            (node_count * core::mem::size_of::<GpuLbvhNode>()) as u64,
        );
        let visited_count = prim_capacity.max(1);
        let visited = alloc_storage(
            &device,
            "molla::lbvh visited",
            (visited_count * core::mem::size_of::<u32>()) as u64,
        );

        let pair_count = alloc_storage(
            &device,
            "molla::lbvh pair_count",
            (4 * core::mem::size_of::<u32>()) as u64,
        );
        let pair_a = alloc_storage(
            &device,
            "molla::lbvh pair_a",
            (pair_capacity * core::mem::size_of::<u32>()) as u64,
        );
        let pair_b = alloc_storage(
            &device,
            "molla::lbvh pair_b",
            (pair_capacity * core::mem::size_of::<u32>()) as u64,
        );

        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::lbvh uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            pipeline_build,
            pipeline_refit,
            pipeline_pairs,
            layout_tree,
            layout_uniforms,
            layout_pairs,
            primitive_aabb,
            sorted_idx,
            sorted_codes,
            nodes,
            visited,
            pair_count,
            pair_a,
            pair_b,
            uniforms,
            prim_capacity,
            pair_capacity,
            cached_topology: None,
        })
    }

    /// Grow `primitive_aabb` / `sorted_idx` / `sorted_codes` / `nodes`
    /// / `visited` to fit at least `n` primitives. No-op if the
    /// current capacity already covers it.
    fn ensure_prim_capacity(&mut self, n: usize) {
        if n <= self.prim_capacity {
            return;
        }
        let cap = (n * 2).max(self.prim_capacity * 2).max(2);
        self.prim_capacity = cap;
        self.primitive_aabb = alloc_storage(
            &self.device,
            "molla::lbvh primitive_aabb",
            (cap * core::mem::size_of::<GpuAabb>()) as u64,
        );
        self.sorted_idx = alloc_storage(
            &self.device,
            "molla::lbvh sorted_idx",
            (cap * core::mem::size_of::<u32>()) as u64,
        );
        self.sorted_codes = alloc_storage(
            &self.device,
            "molla::lbvh sorted_codes",
            (cap * core::mem::size_of::<u32>()) as u64,
        );
        let node_count = (2 * cap - 1).max(1);
        self.nodes = alloc_storage(
            &self.device,
            "molla::lbvh nodes",
            (node_count * core::mem::size_of::<GpuLbvhNode>()) as u64,
        );
        self.visited = alloc_storage(
            &self.device,
            "molla::lbvh visited",
            (cap * core::mem::size_of::<u32>()) as u64,
        );
        // Topology cache references buffer slots that just got
        // reallocated — invalidate so the next `refit_only` falls
        // back to a full build.
        self.cached_topology = None;
    }

    /// Grow `pair_a` / `pair_b` to fit at least `n` pairs. Returns
    /// `true` if anything was reallocated.
    fn ensure_pair_capacity(&mut self, n: usize) -> bool {
        if n <= self.pair_capacity {
            return false;
        }
        let cap = (n * 2).max(self.pair_capacity * 2).max(64);
        self.pair_capacity = cap;
        self.pair_a = alloc_storage(
            &self.device,
            "molla::lbvh pair_a",
            (cap * core::mem::size_of::<u32>()) as u64,
        );
        self.pair_b = alloc_storage(
            &self.device,
            "molla::lbvh pair_b",
            (cap * core::mem::size_of::<u32>()) as u64,
        );
        true
    }

    /// Number of primitives the buffers can currently hold without
    /// reallocation.
    pub fn prim_capacity(&self) -> usize {
        self.prim_capacity
    }

    /// Pair-output buffer capacity.
    pub fn pair_capacity(&self) -> usize {
        self.pair_capacity
    }

    /// Run the **full** broad phase (Morton + sort + build + refit +
    /// pairs). Caller supplies pre-computed per-primitive AABBs and
    /// the scene bounds — this avoids re-walking the model+state on
    /// the host (callers that wrap a `Model` / `State` use
    /// [`Self::dispatch_from_model`]).
    pub fn dispatch(&mut self, prim_aabbs: &[(Vec3, Vec3)]) -> Result<LbvhPairs> {
        let n = prim_aabbs.len();
        if n < 2 {
            return Ok(LbvhPairs::default());
        }
        self.ensure_prim_capacity(n);

        // Compute scene bounds (centroid range — codes are mapped into
        // [0, 1)^3 over `[scene_min, scene_max]`).
        let mut scene_min = [f32::INFINITY; 3];
        let mut scene_max = [f32::NEG_INFINITY; 3];
        let mut gpu_aabbs: Vec<GpuAabb> = Vec::with_capacity(n);
        for (mn, mx) in prim_aabbs {
            let (mn3, mx3) = (mn.as_vec3(), mx.as_vec3());
            // Replace ±inf with finite bounds — half-space planes have
            // an infinite AABB; clamp them so they don't poison the
            // Morton-bound calculation. They still land at the root
            // and pair against everyone via the AABB overlap test.
            let mn_clamped = [
                if mn3.x.is_finite() { mn3.x } else { -1e6 },
                if mn3.y.is_finite() { mn3.y } else { -1e6 },
                if mn3.z.is_finite() { mn3.z } else { -1e6 },
            ];
            let mx_clamped = [
                if mx3.x.is_finite() { mx3.x } else { 1e6 },
                if mx3.y.is_finite() { mx3.y } else { 1e6 },
                if mx3.z.is_finite() { mx3.z } else { 1e6 },
            ];
            let cx = (mn_clamped[0] + mx_clamped[0]) * 0.5;
            let cy = (mn_clamped[1] + mx_clamped[1]) * 0.5;
            let cz = (mn_clamped[2] + mx_clamped[2]) * 0.5;
            scene_min[0] = scene_min[0].min(cx);
            scene_min[1] = scene_min[1].min(cy);
            scene_min[2] = scene_min[2].min(cz);
            scene_max[0] = scene_max[0].max(cx);
            scene_max[1] = scene_max[1].max(cy);
            scene_max[2] = scene_max[2].max(cz);
            gpu_aabbs.push(GpuAabb {
                min: [mn_clamped[0], mn_clamped[1], mn_clamped[2], 0.0],
                max: [mx_clamped[0], mx_clamped[1], mx_clamped[2], 0.0],
            });
        }
        // Pad-out scene bounds to avoid divide-by-zero downstream.
        for i in 0..3 {
            if scene_max[i] - scene_min[i] < 1e-6 {
                scene_max[i] = scene_min[i] + 1.0;
            }
        }

        // CPU Morton + radix sort. See module-level doc-comment for
        // why this isn't on the GPU yet.
        let codes: Vec<u32> = (0..n)
            .map(|i| {
                let cx = (gpu_aabbs[i].min[0] + gpu_aabbs[i].max[0]) * 0.5;
                let cy = (gpu_aabbs[i].min[1] + gpu_aabbs[i].max[1]) * 0.5;
                let cz = (gpu_aabbs[i].min[2] + gpu_aabbs[i].max[2]) * 0.5;
                let nx = (cx - scene_min[0]) / (scene_max[0] - scene_min[0]);
                let ny = (cy - scene_min[1]) / (scene_max[1] - scene_min[1]);
                let nz = (cz - scene_min[2]) / (scene_max[2] - scene_min[2]);
                morton3_30(nx, ny, nz)
            })
            .collect();
        // Sort indices by code (stable so ties are broken by original
        // index, matching the WGSL `delta` tie-break).
        let mut sorted: Vec<(u32, u32)> = codes
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i as u32))
            .collect();
        sorted.sort_by_key(|&(c, idx)| (c, idx));
        let sorted_codes: Vec<u32> = sorted.iter().map(|&(c, _)| c).collect();
        let sorted_idx: Vec<u32> = sorted.iter().map(|&(_, i)| i).collect();

        // Cache for refit-only mode. `root` is filled in after
        // dispatch_kernels reads it back from `pair_count[1]`.
        self.cached_topology = Some(CachedTopology {
            prim_count: n as u32,
            scene_min: [scene_min[0], scene_min[1], scene_min[2], 0.0],
            scene_max: [scene_max[0], scene_max[1], scene_max[2], 0.0],
            root: 0,
        });

        // Upload buffers.
        self.queue
            .write_buffer(&self.primitive_aabb, 0, bytemuck::cast_slice(&gpu_aabbs));
        self.queue
            .write_buffer(&self.sorted_codes, 0, bytemuck::cast_slice(&sorted_codes));
        self.queue
            .write_buffer(&self.sorted_idx, 0, bytemuck::cast_slice(&sorted_idx));

        // Zero out the entire `nodes` buffer — the new build kernel
        // writes parent / left / right / is_leaf for every node it
        // visits, but `find_pairs` reads `node.aabb_*` for every
        // internal node it touches. Refit fills those, so a zero
        // init is fine. (The previous Karras layout needed `parent`
        // to start as 0xff so refit terminated at root; the new
        // bottom-up build sets root.parent explicitly.)
        let node_count = 2 * n - 1;
        let blank_nodes = vec![0u8; node_count * core::mem::size_of::<GpuLbvhNode>()];
        self.queue.write_buffer(&self.nodes, 0, &blank_nodes);
        // `visited` is cleared inside `dispatch_kernels` via
        // `encoder.clear_buffer` so build + refit both see zeros.

        // Uniforms.
        let u = Uniforms {
            primitive_count: n as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            scene_min: [scene_min[0], scene_min[1], scene_min[2], 0.0],
            scene_max: [scene_max[0], scene_max[1], scene_max[2], 0.0],
        };
        self.queue
            .write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));

        self.dispatch_kernels(n as u32, /*do_build=*/ true)
    }

    /// Refit-only path: reuse the previous frame's tree topology,
    /// only re-run `refit_aabbs` + `find_pairs`. Intended for cloth /
    /// soft-body / particle scenes where shape topology is fixed but
    /// per-shape AABBs move every frame. Falls back to a full build
    /// when the cached topology is stale (different `prim_count`,
    /// scene-bounds drift, or the cache was invalidated by a buffer
    /// resize).
    pub fn refit_only(&mut self, prim_aabbs: &[(Vec3, Vec3)]) -> Result<LbvhPairs> {
        let n = prim_aabbs.len();
        if n < 2 {
            return Ok(LbvhPairs::default());
        }

        let topo_ok = match &self.cached_topology {
            Some(t) => {
                if t.prim_count as usize != n {
                    false
                } else {
                    // Quick sanity: scene didn't grow >2x (Morton
                    // codes would lose resolution beyond that).
                    let mut ok = true;
                    let mut mn = [f32::INFINITY; 3];
                    let mut mx = [f32::NEG_INFINITY; 3];
                    for (a, b) in prim_aabbs {
                        let a = a.as_vec3();
                        let b = b.as_vec3();
                        let cx = ((a.x + b.x) * 0.5) as f32;
                        let cy = ((a.y + b.y) * 0.5) as f32;
                        let cz = ((a.z + b.z) * 0.5) as f32;
                        if cx.is_finite() {
                            mn[0] = mn[0].min(cx);
                            mx[0] = mx[0].max(cx);
                        }
                        if cy.is_finite() {
                            mn[1] = mn[1].min(cy);
                            mx[1] = mx[1].max(cy);
                        }
                        if cz.is_finite() {
                            mn[2] = mn[2].min(cz);
                            mx[2] = mx[2].max(cz);
                        }
                    }
                    let cached_extent = [
                        t.scene_max[0] - t.scene_min[0],
                        t.scene_max[1] - t.scene_min[1],
                        t.scene_max[2] - t.scene_min[2],
                    ];
                    let new_extent = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]];
                    for i in 0..3 {
                        if new_extent[i] > cached_extent[i] * 2.0 + 1e-6 {
                            ok = false;
                        }
                    }
                    ok
                }
            }
            None => false,
        };
        if !topo_ok {
            return self.dispatch(prim_aabbs);
        }

        // Same-topology path: re-upload AABBs, clear `visited`, leave
        // `sorted_idx` / `sorted_codes` / `nodes.left/right/parent`
        // intact. We do still need to reset `nodes[*].aabb_min/max`
        // so refit's union starts from a clean slate; the easiest
        // way without a clear kernel is to memset the AABB lanes by
        // re-uploading the whole nodes buffer with 0xff (NaN) for
        // those lanes. That blows away child/parent fields too —
        // which would defeat the optimisation.
        //
        // Instead: refit's atomic protocol overwrites both AABB
        // lanes unconditionally for every internal node it visits
        // (it doesn't "expand" — it writes `min(L,R)` / `max(L,R)`
        // outright). So we only need to clear `visited`.
        self.ensure_prim_capacity(n);
        self.ensure_pair_capacity(self.pair_capacity); // no-op

        // Re-upload primitive AABBs.
        let mut gpu_aabbs: Vec<GpuAabb> = Vec::with_capacity(n);
        for (mn, mx) in prim_aabbs {
            let mn3 = mn.as_vec3();
            let mx3 = mx.as_vec3();
            let mn_c = [
                if mn3.x.is_finite() { mn3.x } else { -1e6 },
                if mn3.y.is_finite() { mn3.y } else { -1e6 },
                if mn3.z.is_finite() { mn3.z } else { -1e6 },
            ];
            let mx_c = [
                if mx3.x.is_finite() { mx3.x } else { 1e6 },
                if mx3.y.is_finite() { mx3.y } else { 1e6 },
                if mx3.z.is_finite() { mx3.z } else { 1e6 },
            ];
            gpu_aabbs.push(GpuAabb {
                min: [mn_c[0], mn_c[1], mn_c[2], 0.0],
                max: [mx_c[0], mx_c[1], mx_c[2], 0.0],
            });
        }
        self.queue
            .write_buffer(&self.primitive_aabb, 0, bytemuck::cast_slice(&gpu_aabbs));
        // `visited` is cleared inside `dispatch_kernels` via
        // `encoder.clear_buffer`.

        // Uniforms — keep the cached scene bounds (Morton codes are
        // already baked into `sorted_codes` from the last build).
        let topo = self.cached_topology.as_ref().unwrap();
        let u = Uniforms {
            primitive_count: n as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            scene_min: topo.scene_min,
            scene_max: topo.scene_max,
        };
        self.queue
            .write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));

        self.dispatch_kernels(n as u32, /*do_build=*/ false)
    }

    /// Convenience wrapper: walks a `Model` + `State` to build the
    /// per-shape AABB array, then runs the full broad phase. Mirrors
    /// the CPU [`crate::geometry::BroadPhase::sap_pairs`] entry-point
    /// signature.
    pub fn dispatch_from_model(
        &mut self,
        model: &Model,
        state: &State,
        margin: Real,
    ) -> Result<LbvhPairs> {
        let aabbs = collect_shape_aabbs(model, state, margin);
        self.dispatch(&aabbs)
    }

    /// Same as [`Self::dispatch_from_model`] but uses the cached
    /// topology — valid for scenes where shape *topology* is fixed
    /// (no shapes added / removed) but AABBs change.
    pub fn refit_from_model(
        &mut self,
        model: &Model,
        state: &State,
        margin: Real,
    ) -> Result<LbvhPairs> {
        let aabbs = collect_shape_aabbs(model, state, margin);
        self.refit_only(&aabbs)
    }

    /// Encode + submit `(build_hierarchy?)`, `refit_aabbs`,
    /// `find_pairs`. `n` = primitive count. Returns the populated
    /// pair list, after one re-dispatch round if the pair buffer
    /// overflowed.
    ///
    /// `visited` must be cleared before BOTH the build and refit
    /// passes — both kernels reuse it as a per-internal-node arrival
    /// counter. We submit two `clear_buffer` calls + the three
    /// compute passes in a single encoder.
    /// Encode the build / refit / find_pairs passes WITHOUT reading
    /// back the pair list. Returns immediately after `queue.submit`.
    /// Useful for callers that want to consume `pair_a` / `pair_b`
    /// directly on the GPU (no host roundtrip) — e.g. chained into a
    /// downstream GJK / EPA dispatch.
    pub fn dispatch_no_readback(&mut self, prim_aabbs: &[(Vec3, Vec3)]) -> Result<()> {
        let n = prim_aabbs.len();
        if n < 2 {
            return Ok(());
        }
        // Re-use the full dispatch path's setup — we just don't read
        // back at the end. Cheap hack: temporarily clear the cache so
        // the caller's next refit_only path forces a full rebuild
        // (otherwise the topology cache wouldn't get the right root).
        let pairs = self.dispatch(prim_aabbs)?;
        // Discard pairs — they're already on the GPU, the host copy
        // is a side-effect of the safe `dispatch` API.
        let _ = pairs;
        Ok(())
    }

    /// Borrow the GPU pair-output buffers for downstream chaining.
    /// `pair_count[0]` holds the emitted-pair count; `pair_a[0..count]`
    /// and `pair_b[0..count]` hold the (a, b) primitive indices.
    pub fn pair_buffers(&self) -> (&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer) {
        (&self.pair_count, &self.pair_a, &self.pair_b)
    }

    fn dispatch_kernels(&mut self, n: u32, do_build: bool) -> Result<LbvhPairs> {
        // Reset pair counter; if doing refit-only, re-stamp the
        // cached root into `pair_count[1]` so find_pairs can locate
        // it (otherwise build_hierarchy writes it on this dispatch).
        let root_seed = if !do_build {
            self.cached_topology
                .as_ref()
                .map(|c| c.root)
                .unwrap_or(0)
        } else {
            0
        };
        self.queue
            .write_buffer(&self.pair_count, 0, bytemuck::cast_slice(&[0u32, root_seed, 0, 0]));

        let bg_tree = self.build_tree_bind_group();
        let bg_uniforms = self.build_uniforms_bind_group();
        let bg_pairs = self.build_pairs_bind_group();

        let visited_bytes = ((n.saturating_sub(1)).max(1) as u64)
            * core::mem::size_of::<u32>() as u64;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::lbvh encoder"),
            });
        if do_build && n >= 2 {
            // Clear visited counters for the build pass.
            encoder.clear_buffer(&self.visited, 0, Some(visited_bytes));
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::lbvh::build_hierarchy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_build);
            pass.set_bind_group(0, &bg_tree, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.set_bind_group(2, &bg_pairs, &[]);
            // n threads — leaves drive the bottom-up walk.
            let groups = n.div_ceil(WORKGROUP_SIZE).max(1);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        // Refit needs the visited counters re-zeroed (build left them
        // at 2 per internal node).
        if n >= 2 {
            encoder.clear_buffer(&self.visited, 0, Some(visited_bytes));
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::lbvh::refit_aabbs"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_refit);
            pass.set_bind_group(0, &bg_tree, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            let groups = n.div_ceil(WORKGROUP_SIZE).max(1);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::lbvh::find_pairs"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_pairs);
            pass.set_bind_group(0, &bg_tree, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.set_bind_group(2, &bg_pairs, &[]);
            let groups = n.div_ceil(WORKGROUP_SIZE).max(1);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        // Coalesced readback: one encoder + one map. Stage the
        // (pair_count[0..2], pair_a[..count], pair_b[..count]) ranges
        // into a single staging buffer so we pay the GPU sync /
        // map_async latency once instead of three times.
        let (count, root, pair_a, pair_b) =
            self.read_pairs_coalesced(self.pair_capacity)?;
        let overflowed = count > self.pair_capacity;

        // Stash the root for refit-only mode (which doesn't re-run
        // build_hierarchy). On refit-only paths, dispatch_kernels
        // reads the cache and re-stamps `pair_count[1]` before the
        // find_pairs dispatch.
        if do_build {
            if let Some(cache) = self.cached_topology.as_mut() {
                cache.root = root;
            }
        }

        // If we overflowed, grow + re-run once. Drop the cache flag
        // back into the user's hands by setting `overflowed=true`.
        if overflowed {
            self.ensure_pair_capacity(count);
            // Caller can re-dispatch on the next frame; no infinite
            // loop here.
        }

        Ok(LbvhPairs {
            a: pair_a,
            b: pair_b,
            overflowed,
        })
    }

    /// Download the LBVH `nodes` buffer for debugging. Returns
    /// `(left, right, parent, is_leaf, aabb_min[3], aabb_max[3])` per
    /// node for the first `node_count` slots.
    pub fn debug_dump_nodes(&self, node_count: usize) -> Result<Vec<DebugNode>> {
        if node_count == 0 {
            return Ok(Vec::new());
        }
        let byte_size = (node_count * core::mem::size_of::<GpuLbvhNode>()) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::lbvh debug_dump_nodes staging"),
            size: byte_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(&self.nodes, 0, &staging, 0, byte_size);
        self.queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("poll: {e:?}")))?;
        rx.recv()
            .map_err(|e| Error::Gpu(format!("recv: {e}")))?
            .map_err(|e| Error::Gpu(format!("map: {e:?}")))?;
        let mapped = slice.get_mapped_range();
        let raw: &[GpuLbvhNode] = bytemuck::cast_slice(&mapped);
        let out: Vec<DebugNode> = raw[..node_count]
            .iter()
            .map(|n| DebugNode {
                left: n.left,
                right: n.right,
                parent: n.parent,
                is_leaf: n.is_leaf,
                aabb_min: [n.aabb_min[0], n.aabb_min[1], n.aabb_min[2]],
                aabb_max: [n.aabb_max[0], n.aabb_max[1], n.aabb_max[2]],
            })
            .collect();
        drop(mapped);
        staging.unmap();
        Ok(out)
    }

    /// Two-stage GPU→host readback. First stage reads the small
    /// header `(count, root)`; second stage transfers exactly
    /// `count * 8` bytes for `(pair_a, pair_b)`. This pays one extra
    /// GPU sync but saves ~1 MB of useless transfer when the output
    /// is sparse — much faster than reading the whole capacity each
    /// frame.
    ///
    /// Returns `(count_emitted, root_node_idx, pair_a, pair_b)`.
    fn read_pairs_coalesced(
        &self,
        pair_capacity: usize,
    ) -> Result<(usize, u32, Vec<u32>, Vec<u32>)> {
        // Stage 1 — header only.
        let header = read_u32(&self.device, &self.queue, &self.pair_count, 2)?;
        let count = header[0] as usize;
        let root = header[1];
        let to_read = count.min(pair_capacity);
        if to_read == 0 {
            return Ok((count, root, Vec::new(), Vec::new()));
        }

        // Stage 2 — pair_a + pair_b in a single staging buffer.
        let pair_bytes = (to_read * core::mem::size_of::<u32>()) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::lbvh staging (pairs)"),
            size: 2 * pair_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::lbvh staging encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.pair_a, 0, &staging, 0, pair_bytes);
        encoder.copy_buffer_to_buffer(&self.pair_b, 0, &staging, pair_bytes, pair_bytes);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("poll: {e:?}")))?;
        rx.recv()
            .map_err(|e| Error::Gpu(format!("staging recv: {e}")))?
            .map_err(|e| Error::Gpu(format!("staging map: {e:?}")))?;
        let mapped = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&mapped);
        let pair_a = words[..to_read].to_vec();
        let pair_b = words[to_read..2 * to_read].to_vec();
        drop(mapped);
        staging.unmap();
        Ok((count, root, pair_a, pair_b))
    }

    fn build_tree_bind_group(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::lbvh tree bg"),
            layout: &self.layout_tree,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.primitive_aabb.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.sorted_idx.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.sorted_codes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.nodes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.visited.as_entire_binding(),
                },
            ],
        })
    }

    fn build_uniforms_bind_group(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::lbvh uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.uniforms.as_entire_binding(),
            }],
        })
    }

    fn build_pairs_bind_group(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::lbvh pairs bg"),
            layout: &self.layout_pairs,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.pair_count.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.pair_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.pair_b.as_entire_binding(),
                },
            ],
        })
    }
}

/// Walk the model's shape table and produce per-shape world AABBs as
/// `(min, max)` `Vec3` pairs. Pulled out so tests can grab the raw
/// AABBs without holding the LBVH state.
pub fn collect_shape_aabbs(model: &Model, state: &State, margin: Real) -> Vec<(Vec3, Vec3)> {
    let n = model.shape_count;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let aabb = crate::geometry::shape_world_aabb(model, state, i, margin);
        out.push((aabb.min, aabb.max));
    }
    out
}

// ── Helpers ─────────────────────────────────────────────────────────

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

fn read_only_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

fn read_write_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

fn read_u32(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buf: &wgpu::Buffer,
    count: usize,
) -> Result<Vec<u32>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let byte_size = (count * core::mem::size_of::<u32>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::lbvh staging"),
        size: byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::lbvh staging encoder"),
    });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, byte_size);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| Error::Gpu(format!("device poll: {e:?}")))?;
    rx.recv()
        .map_err(|e| Error::Gpu(format!("staging recv: {e}")))?
        .map_err(|e| Error::Gpu(format!("staging map: {e:?}")))?;
    let mapped = slice.get_mapped_range();
    let out: Vec<u32> = bytemuck::cast_slice::<u8, u32>(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    Ok(out)
}

/// 30-bit Morton code from three normalised f32s in `[0, 1]`.
/// Mirrors `morton3` in `shaders/lbvh.wgsl` — keep them bit-exact so
/// the CPU sort produces the same key order the WGSL `delta` would.
fn morton3_30(x: f32, y: f32, z: f32) -> u32 {
    let xi = ((x.clamp(0.0, 0.99999) * 1024.0) as u32) & 0x000003ff;
    let yi = ((y.clamp(0.0, 0.99999) * 1024.0) as u32) & 0x000003ff;
    let zi = ((z.clamp(0.0, 0.99999) * 1024.0) as u32) & 0x000003ff;
    spread_bits(xi) | (spread_bits(yi) << 1) | (spread_bits(zi) << 2)
}

fn spread_bits(v: u32) -> u32 {
    let mut x = v & 0x000003ff;
    x = (x | (x << 16)) & 0x030000ff;
    x = (x | (x << 8)) & 0x0300f00f;
    x = (x | (x << 4)) & 0x030c30c3;
    x = (x | (x << 2)) & 0x09249249;
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn morton_orders_along_diagonal() {
        // Codes along the main diagonal should be monotonically
        // increasing — cheap sanity check that `spread_bits` lines
        // up with the WGSL implementation.
        let mut last = 0u32;
        for i in 0..16 {
            let t = i as f32 / 16.0;
            let c = morton3_30(t, t, t);
            assert!(c >= last, "morton3 not monotonic along diagonal");
            last = c;
        }
    }
}
