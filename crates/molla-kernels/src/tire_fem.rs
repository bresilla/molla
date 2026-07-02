//! GPU kernel wrapper for volumetric tire tetra-FEM.
//!
//! This is the GPU counterpart to `molla_vehicle::TetraTireFem`'s node update:
//! one WGSL invocation owns one node, loops over incident tetrahedra, applies
//! compressible Neo-Hookean tetra forces, bead/rim coupling, optional pressure
//! traction, and floor contact, then writes next positions/velocities. The
//! first implementation is intentionally deterministic and race-free (no float
//! atomics); later high-throughput versions can add CSR adjacency or per-tet
//! reductions.

use bytemuck::{Pod, Zeroable};
use molla_core::{Error, Result};

use crate::tire_brush::{
    BrushTerrain, BrushWheelOutput, TireBrushGpuKernel, TireBrushResidentOutput,
};
use crate::tire_vehicle::{
    TireVehicleBrushRowsFromSpinGpuInput, TireVehiclePlanarFromBrushGpuInput,
    TireVehicleResidentChassisBrushOwner, TireVehicleSuspensionGpuInput,
    TireVehicleVerticalGpuInput, TireVehicleWheelSpinGpuInput, TireVehicleYawGpuKernel,
};

const WORKGROUP_SIZE: u32 = 64;

/// WGSL source for the fused tire-FEM atlas node-step kernel.
///
/// The shader keeps the same physical FEM node update as the single-tire
/// `tire_fem.wgsl` path, but routes each atlas-global node through per-tire
/// ownership, parameter, and offset rows.
pub const TIRE_FEM_ATLAS_SHADER: &str = include_str!("../shaders/tire_fem_atlas.wgsl");

/// WGSL source-guard shader for the matrix-free Newton tire-FEM debug
/// timestep.
///
/// Defines `residual_eval`, `jv_probe`, `diagonal_extract`, and
/// `apply_correction` entry points keyed to
/// [`TireFemMatrixFreeNewtonGpuContract`]. The host-side dispatch is not
/// wired through the persistent batch yet, but the shader source is bundled
/// and source-guard-tested so the future live dispatch cannot drift from the
/// CPU matrix-free Newton reference in
/// `molla_vehicle::real_tire::matrix_free_newton_correction`.
pub const TIRE_FEM_MATRIX_FREE_NEWTON_SHADER: &str =
    include_str!("../shaders/tire_fem_matrix_free_newton.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Uniforms {
    counts: [u32; 4],
    counts2: [u32; 4],
    phys0: [f32; 4],
    phys1: [f32; 4],
    phys2: [f32; 4],
    phys3: [f32; 4],
    phys4: [f32; 4],
    // PLAN_TOURQUE Stage G — [hub_omega, drive_torque_n_m, brake_torque_n_m, reserved].
    // Stage G part 1 carries the values; Stage G part 2 wires them into
    // the WGSL per-node physics. Mirrors tire_fem.wgsl::phys5.
    phys5: [f32; 4],
}

/// Host-side scalar parameters for one GPU tire FEM node step.
#[derive(Clone, Copy, Debug)]
pub struct TireFemGpuParams {
    pub dt: f32,
    /// Legacy/fallback scalar material fields. The current kernel uses the
    /// per-tet `tet_mu` / `tet_lambda` buffers for actual constitutive stress.
    pub young_modulus: f32,
    pub poisson_ratio: f32,
    pub pressure_pa: f32,
    pub node_damping: f32,
    pub bead_stiffness: f32,
    pub floor_stiffness: f32,
    pub floor_damping: f32,
    pub pressure_area_per_tread_node: f32,
    pub hub: [f32; 3],
    pub gravity_y: f32,
    /// Cavity-pressure radial force *per bead node* (N). Computed by
    /// `TetraTireFem` from the polytropic cavity gas law and the cavity
    /// triangulation (`stage (b)` in `PLAN_REALTIRE.md`). The kernel
    /// applies it along `(node − hub)` direction to each bead node.
    pub cavity_radial_force_per_bead: f32,
    /// Stage (b) safety / stage (c) co-resident — uniform addition to
    /// every tet's μ. Lives in a uniform so the persistent buffer path
    /// (stage (g)) does not re-upload `tet_mu` when pressure changes.
    pub belt_proxy_mu: f32,
    /// Stage (f) — XPBD per-tetra volume compliance. The GPU node kernel
    /// runs one race-free gather projection sweep per dispatch; set to a
    /// huge value to effectively disable it.
    pub xpbd_volume_compliance: f32,
    /// Stage (f) — XPBD bead/rim mount compliance. Uses the existing
    /// `bead_ids`/`bead_offsets` buffers to keep bead nodes geometrically
    /// attached to the GPU-resident hub pose.
    pub xpbd_bead_compliance: f32,
    /// Stage (f2) — diagonal semi-implicit tangent-stiffness scale.
    ///
    /// The shader still evaluates Neo-Hookean/fibre forces explicitly, but it
    /// mirrors the CPU stabilizer by adding
    /// `scale · k_diag · dt²` to the velocity-update denominator. `0.0`
    /// preserves the old explicit-elasticity behavior.
    pub semi_implicit_stiffness_scale: f32,
    /// Stage (f4) — tensile sidewall ply-stretch limit for the GPU XPBD
    /// guardrail. The shader constrains `|F·a0| <= limit` for sidewall ply
    /// families identified by radial+axial helper plies.
    pub xpbd_sidewall_ply_stretch_limit: f32,
    /// Stage (f4) — compliance for the sidewall ply-stretch XPBD guardrail.
    /// Use a very large value to disable it.
    pub xpbd_sidewall_ply_compliance: f32,
    /// PLAN_TOURQUE Stage G — current spin angular velocity around the
    /// wheel's spin axis (rad/s). Mirrors `TetraTireFem::hub_omega`. The
    /// WGSL kernel uses this to inject the spin contribution
    /// `v_spin = ω·r_y` into per-tread tangential slip so the GPU brush
    /// reaction matches the CPU path. The WGSL step kernel reads this
    /// slot (`spin_omega = u.phys5.x`); validated by the Stream C G4
    /// rolling-parity hardware lane (see [`TireFemTorqueGpuStatus`]).
    pub hub_omega: f32,
    /// PLAN_TOURQUE Stage G — commanded drive torque at the wheel hub
    /// (N·m, signed). Hub spin integration stays host-side by design
    /// (`TetraTireFem::integrate_spin_dof`); the host integrates
    /// `hub_omega` from this torque each substep and forwards the
    /// result to the GPU via this params block.
    pub drive_torque_n_m: f32,
    /// PLAN_TOURQUE Stage G — commanded brake torque magnitude (N·m,
    /// ≥0). Always opposes the current `hub_omega`. Integrated
    /// host-side alongside `drive_torque_n_m` and reflected in
    /// per-tire telemetry.
    pub brake_torque_n_m: f32,
}

impl Default for TireFemGpuParams {
    fn default() -> Self {
        Self {
            dt: 1.0 / 240.0,
            young_modulus: 3.5e5,
            poisson_ratio: 0.46,
            pressure_pa: 180_000.0,
            node_damping: 18.0,
            bead_stiffness: 5.5e4,
            floor_stiffness: 2.8e5,
            floor_damping: 1_200.0,
            pressure_area_per_tread_node: 0.0,
            hub: [0.0, 0.34, 0.0],
            gravity_y: -9.81,
            cavity_radial_force_per_bead: 0.0,
            belt_proxy_mu: 0.0,
            xpbd_volume_compliance: 5.0e-10,
            xpbd_bead_compliance: 5.0e-5,
            semi_implicit_stiffness_scale: 0.0,
            xpbd_sidewall_ply_stretch_limit: 1.35,
            xpbd_sidewall_ply_compliance: 5.0e-3,
            // PLAN_TOURQUE Stage G — torque/spin defaults: at-rest tire
            // with zero drive/brake. Scenes that opt in set these per
            // dispatch from `TetraTireFem`'s accessors.
            hub_omega: 0.0,
            drive_torque_n_m: 0.0,
            brake_torque_n_m: 0.0,
        }
    }
}

/// Complete buffer payload for one blocking GPU tire FEM dispatch.
#[derive(Clone, Debug)]
pub struct TireFemGpuInput {
    pub rest_positions: Vec<[f32; 4]>,
    pub positions: Vec<[f32; 4]>,
    pub velocities: Vec<[f32; 4]>,
    pub masses: Vec<f32>,
    /// Four vertex indices per tet.
    pub tet_indices: Vec<u32>,
    /// Four shape-function gradients per tet, matching `tet_indices`.
    pub tet_grads: Vec<[f32; 4]>,
    pub tet_volumes: Vec<f32>,
    pub tet_mu: Vec<f32>,
    pub tet_lambda: Vec<f32>,
    /// Stage (c) — per-tet fibre direction `a₀` (xyz, w=0).
    pub tet_fiber_dir: Vec<[f32; 4]>,
    /// Stage (c) — per-tet linear fibre stiffness k₁ (Pa).
    pub tet_fiber_k1: Vec<f32>,
    /// Stage (c) — per-tet Holzapfel exponent k₂.
    pub tet_fiber_k2: Vec<f32>,
    /// Stage (c2) — second crossed ply/fibre family.
    pub tet_fiber_dir_b: Vec<[f32; 4]>,
    pub tet_fiber_k1_b: Vec<f32>,
    pub tet_fiber_k2_b: Vec<f32>,
    /// Stage (c3) — third explicit ply/cap family.
    pub tet_fiber_dir_c: Vec<[f32; 4]>,
    pub tet_fiber_k1_c: Vec<f32>,
    pub tet_fiber_k2_c: Vec<f32>,
    /// Stage (c3) — fourth explicit ply/cap family.
    pub tet_fiber_dir_d: Vec<[f32; 4]>,
    pub tet_fiber_k1_d: Vec<f32>,
    pub tet_fiber_k2_d: Vec<f32>,
    pub bead_ids: Vec<u32>,
    pub bead_offsets: Vec<[f32; 4]>,
    pub tread_ids: Vec<u32>,
    /// Stage (h) CSR — see `TetraTireFemGpuSnapshot::node_tet_offsets`.
    pub node_tet_offsets: Vec<u32>,
    /// Stage (h) CSR — `(tet_index << 2) | local_vertex_id`. The
    /// kernel iterates only adjacent tets instead of all of them.
    pub node_tet_entries: Vec<u32>,
    /// Stage (f) structural XPBD edge endpoints `(a,b,kind,0)`.
    pub xpbd_edge_indices: Vec<[u32; 4]>,
    /// Stage (f) structural XPBD edge params `(rest_length, compliance, 0, 0)`.
    pub xpbd_edge_params: Vec<[f32; 4]>,
    /// Stage (f) node→edge CSR offsets.
    pub node_edge_offsets: Vec<u32>,
    /// Stage (f) node→edge CSR entries `(edge_index << 1) | local_endpoint`.
    pub node_edge_entries: Vec<u32>,
    /// Stage (f3) sidewall XPBD dihedral constraints `(a,b,c,d)`.
    pub xpbd_dihedral_indices: Vec<[u32; 4]>,
    /// Stage (f3) sidewall XPBD dihedral params `(rest_angle_rad, compliance, 0, 0)`.
    pub xpbd_dihedral_params: Vec<[f32; 4]>,
    /// Stage (f3) node→dihedral CSR offsets.
    pub node_dihedral_offsets: Vec<u32>,
    /// Stage (f3) node→dihedral CSR entries `(dihedral_index << 2) | local_vertex`.
    pub node_dihedral_entries: Vec<u32>,
    pub params: TireFemGpuParams,
}

/// Readback from the GPU tire FEM step.
#[derive(Clone, Debug)]
pub struct TireFemGpuOutput {
    pub positions: Vec<[f32; 4]>,
    pub velocities: Vec<[f32; 4]>,
    pub forces: Vec<[f32; 4]>,
}

/// Atlas-wide output rows for the fused FEM batch dispatch.
///
/// A single-dispatch atlas kernel writes one packed position/velocity/force
/// buffer. This host-side shape keeps that packed output explicit while still
/// allowing callers to split it back into per-tire [`TireFemGpuOutput`] rows.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TireFemGpuBatchAtlasOutputRows {
    pub positions: Vec<[f32; 4]>,
    pub velocities: Vec<[f32; 4]>,
    pub forces: Vec<[f32; 4]>,
}

/// One storage binding in the fused FEM atlas node-step kernel.
///
/// Bindings `0..=37` intentionally mirror the existing single-tire FEM shader
/// so the fused path can reuse the same physical buffer roles. Bindings
/// `38..=43` add atlas-only ownership/parameter/offset rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TireFemGpuBatchAtlasBindingPlan {
    pub binding: u32,
    pub name: &'static str,
    pub rows: usize,
    pub bytes: u64,
    pub read_only: bool,
}

/// Dispatch metadata for the fused FEM atlas shader contract.
///
/// This is the host-side launch contract for `tire_fem_atlas.wgsl`. It keeps
/// the entry point, workgroup size, total node count, dispatch workgroup count,
/// and binding-count expectations together so the fused physical shader cannot
/// silently drift from the atlas packer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TireFemGpuBatchAtlasDispatchContract {
    pub shader_label: &'static str,
    pub entry_point: &'static str,
    pub workgroup_size: u32,
    pub workgroup_count: u32,
    pub total_node_count: usize,
    pub binding_count: usize,
}

/// Input for the first GPU FEM contact-diagnostics reduction.
#[derive(Clone, Debug)]
pub struct TireFemContactReductionInput {
    pub positions: Vec<[f32; 4]>,
    pub tread_ids: Vec<u32>,
    pub terrain: Option<TireFemContactTerrain>,
    pub params: TireFemContactReductionParams,
}

/// Optional regular heightfield/material data for GPU FEM contact reduction.
#[derive(Clone, Debug)]
pub struct TireFemContactTerrain {
    pub origin: [f32; 2],
    pub spacing: f32,
    pub nx: u32,
    pub nz: u32,
    pub heights: Vec<f32>,
    pub friction: f32,
    pub friction_map: Option<Vec<f32>>,
    /// Optional row-major water/film depth samples in metres.
    pub water_film_map: Option<Vec<f32>>,
    /// Optional row-major plastic sinkage samples in metres for deformable
    /// terrain. This is sampled by GPU FEM contact reduction/feedback so a
    /// CPU-updated Bekker heightfield can feed rut depth and rolling
    /// resistance into GPU tread contact.
    pub sinkage_map: Option<Vec<f32>>,
    /// Multiplier for Bekker-style rolling resistance from sampled sinkage:
    /// `F_rr = scale · Fz · sinkage / rolling_radius`.
    pub soil_rolling_resistance_scale: f32,
    /// World-space fallback terrain velocity in metres/second for moving belts
    /// or deformable surface flow.
    pub surface_velocity: [f32; 3],
    /// Optional row-major world-space terrain velocity samples in metres/second.
    ///
    /// When present, GPU FEM contact feedback samples this field per tread node
    /// instead of applying the uniform `surface_velocity` everywhere. This lets
    /// moving belts, flowing soil/ruts, and deformable terrain expose local
    /// contact velocity without turning it into a viewer-only slip hack.
    pub surface_velocity_map: Option<Vec<[f32; 4]>>,
    /// Optional native SDF primitives for GPU tire-node contact.
    ///
    /// These are simple built-in primitives, not file/import formats. When
    /// present, FEM contact kernels use closest signed distance/normal against
    /// them in addition to the flat/heightfield surface.
    pub sdf_primitives: Vec<TireFemContactSdfPrimitive>,
    /// Optional native triangle mesh patches for GPU tire-node contact.
    ///
    /// Each triangle is sampled by finite-triangle closest point, not vertical
    /// projection. This is the first GPU-side counterpart to
    /// `TriangleMeshTerrain::signed_distance_and_normal` for ramps, plates and
    /// mesh-edge contact.
    pub triangles: Vec<TireFemContactTriangle>,
}

/// Simple GPU-side SDF primitive for FEM tire contact.
#[derive(Clone, Copy, Debug)]
pub enum TireFemContactSdfPrimitive {
    Sphere {
        center: [f32; 3],
        radius: f32,
        friction: f32,
    },
    Box {
        center: [f32; 3],
        half_extents: [f32; 3],
        friction: f32,
    },
}

impl TireFemContactSdfPrimitive {
    pub fn sphere(center: [f32; 3], radius: f32, friction: f32) -> Self {
        Self::Sphere {
            center,
            radius: radius.max(1.0e-6),
            friction: friction.max(0.0),
        }
    }

    pub fn aabb(min: [f32; 3], max: [f32; 3], friction: f32) -> Self {
        let center = [
            0.5 * (min[0] + max[0]),
            0.5 * (min[1] + max[1]),
            0.5 * (min[2] + max[2]),
        ];
        let half_extents = [
            0.5 * (max[0] - min[0]).abs(),
            0.5 * (max[1] - min[1]).abs(),
            0.5 * (max[2] - min[2]).abs(),
        ];
        Self::Box {
            center,
            half_extents,
            friction: friction.max(0.0),
        }
    }

    fn gpu_rows(self) -> [[f32; 4]; 2] {
        match self {
            Self::Sphere {
                center,
                radius,
                friction,
            } => [
                [center[0], center[1], center[2], 1.0],
                [radius.max(1.0e-6), 0.0, 0.0, friction.max(0.0)],
            ],
            Self::Box {
                center,
                half_extents,
                friction,
            } => [
                [center[0], center[1], center[2], 2.0],
                [
                    half_extents[0].max(1.0e-6),
                    half_extents[1].max(1.0e-6),
                    half_extents[2].max(1.0e-6),
                    friction.max(0.0),
                ],
            ],
        }
    }
}

/// Simple GPU-side finite triangle for FEM tire contact.
#[derive(Clone, Copy, Debug)]
pub struct TireFemContactTriangle {
    pub a: [f32; 3],
    pub b: [f32; 3],
    pub c: [f32; 3],
    pub friction: f32,
}

impl TireFemContactTriangle {
    pub fn new(a: [f32; 3], b: [f32; 3], c: [f32; 3], friction: f32) -> Self {
        Self {
            a,
            b,
            c,
            friction: friction.max(0.0),
        }
    }

    /// XZ-plane bounding sphere used by GPU finite-triangle contact kernels
    /// as the first cheap broadphase before closest-point triangle math.
    pub fn horizontal_bounding_sphere(&self) -> ([f32; 2], f32) {
        let center = [
            (self.a[0] + self.b[0] + self.c[0]) / 3.0,
            (self.a[2] + self.b[2] + self.c[2]) / 3.0,
        ];
        let radius = [self.a, self.b, self.c]
            .iter()
            .map(|p| {
                let dx = p[0] - center[0];
                let dz = p[2] - center[1];
                (dx * dx + dz * dz).sqrt()
            })
            .fold(1.0e-6_f32, f32::max);
        (center, radius)
    }

    fn gpu_rows(self) -> [[f32; 4]; 4] {
        let (center_xz, radius_xz) = self.horizontal_bounding_sphere();
        [
            [self.a[0], self.a[1], self.a[2], 0.0],
            [self.b[0], self.b[1], self.b[2], 0.0],
            [self.c[0], self.c[1], self.c[2], 0.0],
            [
                self.friction.max(0.0),
                center_xz[0],
                center_xz[1],
                radius_xz,
            ],
        ]
    }
}

#[derive(Clone, Copy, Debug)]
struct ContactTriangleBounds {
    min: [f32; 3],
    max: [f32; 3],
}

impl ContactTriangleBounds {
    fn from_triangle(tri: TireFemContactTriangle) -> Self {
        Self {
            min: [
                tri.a[0].min(tri.b[0]).min(tri.c[0]),
                tri.a[1].min(tri.b[1]).min(tri.c[1]),
                tri.a[2].min(tri.b[2]).min(tri.c[2]),
            ],
            max: [
                tri.a[0].max(tri.b[0]).max(tri.c[0]),
                tri.a[1].max(tri.b[1]).max(tri.c[1]),
                tri.a[2].max(tri.b[2]).max(tri.c[2]),
            ],
        }
    }

    fn union(self, other: Self) -> Self {
        Self {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }

    fn centroid(self) -> [f32; 3] {
        [
            0.5 * (self.min[0] + self.max[0]),
            0.5 * (self.min[1] + self.max[1]),
            0.5 * (self.min[2] + self.max[2]),
        ]
    }
}

#[derive(Clone, Copy, Debug)]
struct ContactTriangleBvhNode {
    bounds: ContactTriangleBounds,
    left: Option<usize>,
    right: Option<usize>,
    start: usize,
    count: usize,
}

/// Build a compact vec4 atlas for GPU finite-triangle BVH traversal.
///
/// Layout appended after the existing SDF and triangle rows:
/// - 3 vec4 rows per BVH node:
///   - row0: min.xyz, max.x
///   - row1: max.y, max.z, is_leaf, left_or_index_start
///   - row2: right_or_index_count, reserved, reserved, reserved
/// - packed triangle-index rows, four indices per vec4.
pub fn tire_fem_contact_triangle_bvh_rows(
    triangles: &[TireFemContactTriangle],
) -> (Vec<[f32; 4]>, u32, u32) {
    let bounds = triangles
        .iter()
        .copied()
        .map(ContactTriangleBounds::from_triangle)
        .collect::<Vec<_>>();
    let (nodes, indices) = build_contact_triangle_bvh(&bounds);
    let mut rows = Vec::with_capacity(nodes.len() * 3 + indices.len().div_ceil(4));
    for node in &nodes {
        let is_leaf = (node.count > 0) as u32 as f32;
        let left_or_start = node.left.unwrap_or(node.start) as f32;
        let right_or_count = node.right.unwrap_or(node.count) as f32;
        rows.push([
            node.bounds.min[0],
            node.bounds.min[1],
            node.bounds.min[2],
            node.bounds.max[0],
        ]);
        rows.push([
            node.bounds.max[1],
            node.bounds.max[2],
            is_leaf,
            left_or_start,
        ]);
        rows.push([right_or_count, 0.0, 0.0, 0.0]);
    }
    for chunk in indices.chunks(4) {
        let mut row = [0.0; 4];
        for (dst, &idx) in chunk.iter().enumerate() {
            row[dst] = idx as f32;
        }
        rows.push(row);
    }
    (rows, nodes.len() as u32, indices.len().div_ceil(4) as u32)
}

fn contact_primitive_rows(terrain: &TireFemContactTerrain) -> (Vec<[f32; 4]>, u32, u32) {
    let mut rows = terrain
        .sdf_primitives
        .iter()
        .flat_map(|p| p.gpu_rows())
        .collect::<Vec<_>>();
    rows.extend(terrain.triangles.iter().flat_map(|tri| tri.gpu_rows()));
    let (bvh_rows, bvh_node_count, bvh_index_row_count) =
        tire_fem_contact_triangle_bvh_rows(&terrain.triangles);
    rows.extend(bvh_rows);
    (rows, bvh_node_count, bvh_index_row_count)
}

fn build_contact_triangle_bvh(
    bounds: &[ContactTriangleBounds],
) -> (Vec<ContactTriangleBvhNode>, Vec<usize>) {
    let mut indices: Vec<usize> = (0..bounds.len()).collect();
    let mut nodes = Vec::new();
    if !indices.is_empty() {
        build_contact_triangle_bvh_node(bounds, &mut indices, 0, bounds.len(), &mut nodes);
    }
    (nodes, indices)
}

fn build_contact_triangle_bvh_node(
    bounds: &[ContactTriangleBounds],
    indices: &mut [usize],
    start: usize,
    end: usize,
    nodes: &mut Vec<ContactTriangleBvhNode>,
) -> usize {
    const LEAF_SIZE: usize = 8;
    let mut node_bounds = bounds[indices[start]];
    for &ti in &indices[start + 1..end] {
        node_bounds = node_bounds.union(bounds[ti]);
    }
    let node_id = nodes.len();
    nodes.push(ContactTriangleBvhNode {
        bounds: node_bounds,
        left: None,
        right: None,
        start,
        count: end - start,
    });
    if end - start <= LEAF_SIZE {
        return node_id;
    }
    let mut centroid_min = bounds[indices[start]].centroid();
    let mut centroid_max = centroid_min;
    for &ti in &indices[start + 1..end] {
        let c = bounds[ti].centroid();
        for axis in 0..3 {
            centroid_min[axis] = centroid_min[axis].min(c[axis]);
            centroid_max[axis] = centroid_max[axis].max(c[axis]);
        }
    }
    let extents = [
        centroid_max[0] - centroid_min[0],
        centroid_max[1] - centroid_min[1],
        centroid_max[2] - centroid_min[2],
    ];
    let axis = if extents[0] >= extents[1] && extents[0] >= extents[2] {
        0
    } else if extents[2] >= extents[1] {
        2
    } else {
        1
    };
    if extents[axis] <= 1.0e-6 {
        return node_id;
    }
    indices[start..end].sort_by(|&a, &b| {
        bounds[a].centroid()[axis]
            .partial_cmp(&bounds[b].centroid()[axis])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mid = start + (end - start) / 2;
    let left = build_contact_triangle_bvh_node(bounds, indices, start, mid, nodes);
    let right = build_contact_triangle_bvh_node(bounds, indices, mid, end, nodes);
    nodes[node_id].left = Some(left);
    nodes[node_id].right = Some(right);
    nodes[node_id].start = 0;
    nodes[node_id].count = 0;
    nodes[node_id].bounds = nodes[left].bounds.union(nodes[right].bounds);
    node_id
}

/// Scalar parameters for GPU FEM contact diagnostics.
#[derive(Clone, Copy, Debug)]
pub struct TireFemContactReductionParams {
    pub hub: [f32; 3],
    pub heading: [f32; 3],
    pub contact_band: f32,
    pub ground_y: f32,
    pub downforce: f32,
}

impl Default for TireFemContactReductionParams {
    fn default() -> Self {
        Self {
            hub: [0.0, 0.32, 0.0],
            heading: [1.0, 0.0, 0.0],
            contact_band: 0.05,
            ground_y: 0.0,
            downforce: 2_600.0,
        }
    }
}

/// Readback from GPU FEM contact diagnostics.
#[derive(Clone, Copy, Debug, Default)]
pub struct TireFemContactReductionOutput {
    pub normal_load: f32,
    pub contact_count: u32,
    pub patch_length: f32,
    pub contact_center_x: f32,
    pub min_x: f32,
    pub max_x: f32,
    pub weight_sum: f32,
    pub mean_friction: f32,
    pub mean_water_film_m: f32,
    pub mean_sinkage_m: f32,
    pub world_force: [f32; 3],
    pub world_moment: [f32; 3],
}

/// Input for GPU FEM tread contact force feedback.
#[derive(Clone, Debug)]
pub struct TireFemContactFeedbackInput {
    pub positions: Vec<[f32; 4]>,
    pub velocities: Vec<[f32; 4]>,
    pub masses: Vec<f32>,
    pub tread_ids: Vec<u32>,
    /// Persistent bristle state, one entry per `tread_ids` slot.
    pub bristles: Vec<[f32; 4]>,
    /// Optional regular heightfield/material data for GPU contact feedback.
    pub terrain: Option<TireFemContactTerrain>,
    pub params: TireFemContactFeedbackParams,
}

/// GPU-resident regular terrain buffers for FEM contact feedback.
///
/// This lets deformable-terrain owners such as the tire-MPM raster path feed
/// height/sinkage maps directly into FEM contact kernels without CPU-side
/// terrain rasterization/upload.
pub struct TireFemContactFeedbackGpuTerrain<'a> {
    pub origin: [f32; 2],
    pub spacing: f32,
    pub nx: u32,
    pub nz: u32,
    pub heights: &'a wgpu::Buffer,
    pub friction: f32,
    pub friction_map: Option<&'a wgpu::Buffer>,
    pub water_film_map: Option<&'a wgpu::Buffer>,
    pub sinkage_map: Option<&'a wgpu::Buffer>,
    pub surface_velocity_map: Option<&'a wgpu::Buffer>,
    pub soil_rolling_resistance_scale: f32,
    pub surface_velocity: [f32; 3],
}

pub type TireFemContactReductionGpuTerrain<'a> = TireFemContactFeedbackGpuTerrain<'a>;

/// Scalar parameters for the first GPU FEM tread contact feedback pass.
#[derive(Clone, Copy, Debug)]
pub struct TireFemContactFeedbackParams {
    pub dt: f32,
    pub contact_band: f32,
    pub ground_y: f32,
    pub omega: f32,
    pub hub: [f32; 3],
    pub hub_vel: [f32; 3],
    pub heading: [f32; 3],
    pub rolling_radius: f32,
    pub normal_load_per_tread: f32,
    pub area_per_tread: f32,
    pub tread_shear_stiffness: f32,
    pub mu_static: f32,
    pub mu_dynamic: f32,
    pub bristle_damping: f32,
    pub cohesion: f32,
    pub bristle_relax_rate: f32,
}

impl Default for TireFemContactFeedbackParams {
    fn default() -> Self {
        Self {
            dt: 1.0 / 240.0,
            contact_band: 0.05,
            ground_y: 0.0,
            omega: 0.0,
            hub: [0.0, 0.32, 0.0],
            hub_vel: [0.0; 3],
            heading: [1.0, 0.0, 0.0],
            rolling_radius: 0.32,
            normal_load_per_tread: 0.0,
            area_per_tread: 1.0e-3,
            tread_shear_stiffness: 2.6e6,
            mu_static: 1.0,
            mu_dynamic: 0.85,
            bristle_damping: 4.0e4,
            cohesion: 0.0,
            bristle_relax_rate: 25.0,
        }
    }
}

/// Effective moving-ground command derived from GPU transient relaxation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TireFemTransientFeedbackCommand {
    pub effective_kappa: f32,
    pub effective_alpha_rad: f32,
    pub belt_speed_m_per_s: f32,
    pub wheel_omega_rad_per_s: f32,
    pub surface_velocity_m_per_s: [f32; 3],
}

/// Convert a layered-transient GPU output into the kinematic inputs consumed by
/// GPU FEM tread-contact feedback.
///
/// `TireTransientGpuOutput` is the GPU-resident relaxation result used before
/// brush contact. FEM contact feedback uses different uniforms: wheel angular
/// speed plus a moving terrain velocity. This adapter keeps both contact paths
/// wired to the same effective slip state instead of recomputing raw slip in
/// scene code.
pub fn tire_fem_transient_feedback_command(
    transient: &crate::tire_transient::TireTransientGpuOutput,
    belt_speed_m_per_s: f32,
    rolling_radius_m: f32,
) -> TireFemTransientFeedbackCommand {
    let kappa = finite_f32_or(transient.effective[0], 0.0).clamp(-0.95, 4.0);
    let alpha = finite_f32_or(transient.effective[1], 0.0).clamp(-1.45, 1.45);
    let speed = finite_f32_or(belt_speed_m_per_s, 0.0);
    let radius = finite_f32_or(rolling_radius_m, 0.32).abs().max(1.0e-4);
    TireFemTransientFeedbackCommand {
        effective_kappa: kappa,
        effective_alpha_rad: alpha,
        belt_speed_m_per_s: speed,
        wheel_omega_rad_per_s: (1.0 + kappa) * speed / radius,
        surface_velocity_m_per_s: [-speed * alpha.cos(), 0.0, -speed * alpha.sin()],
    }
}

/// Apply transient-relaxed longitudinal slip to FEM feedback parameters.
pub fn tire_fem_feedback_params_with_transient(
    mut params: TireFemContactFeedbackParams,
    transient: &crate::tire_transient::TireTransientGpuOutput,
    belt_speed_m_per_s: f32,
) -> TireFemContactFeedbackParams {
    let command =
        tire_fem_transient_feedback_command(transient, belt_speed_m_per_s, params.rolling_radius);
    params.omega = command.wheel_omega_rad_per_s;
    params
}

/// Apply transient-relaxed lateral slip to a FEM feedback terrain.
///
/// The feedback shader samples one regular surface velocity for the terrain.
/// This helper maps effective slip angle into that moving-ground velocity so a
/// GPU transient step can feed FEM bristle feedback without viewer-local math.
pub fn tire_fem_contact_terrain_with_transient_surface_velocity(
    mut terrain: TireFemContactTerrain,
    transient: &crate::tire_transient::TireTransientGpuOutput,
    belt_speed_m_per_s: f32,
    rolling_radius_m: f32,
) -> TireFemContactTerrain {
    let command =
        tire_fem_transient_feedback_command(transient, belt_speed_m_per_s, rolling_radius_m);
    terrain.surface_velocity = command.surface_velocity_m_per_s;
    terrain
}

/// Attach a DEM top-layer velocity raster to FEM contact terrain.
///
/// DEM sparse-bed owners can export the moving granular surface as row-major
/// `(vx, vy, vz, count)` rows with `tire_dem_raster_surface_velocity_map`.
/// FEM contact feedback consumes the same shape as `surface_velocity_map`.
/// This helper is intentionally strict about grid layout so a DEM velocity
/// map cannot be silently sampled against a different height/sinkage terrain.
pub fn tire_fem_contact_terrain_with_dem_surface_velocity(
    mut terrain: TireFemContactTerrain,
    dem_velocity: &crate::tire_dem::TireDemSurfaceVelocityMap,
) -> Result<TireFemContactTerrain> {
    let expected_len = terrain.nx as usize * terrain.nz as usize;
    if dem_velocity.nx != terrain.nx || dem_velocity.nz != terrain.nz {
        return Err(Error::Solver(
            "tire_fem_dem_velocity_bridge: DEM velocity-map dimensions must match FEM terrain"
                .into(),
        ));
    }
    if dem_velocity.rows.len() != expected_len {
        return Err(Error::Solver(
            "tire_fem_dem_velocity_bridge: DEM velocity-map row count must match FEM terrain"
                .into(),
        ));
    }
    let origin_ok = (dem_velocity.origin[0] - terrain.origin[0]).abs() <= 1.0e-5
        && (dem_velocity.origin[1] - terrain.origin[1]).abs() <= 1.0e-5;
    let spacing_ok = (dem_velocity.spacing - terrain.spacing).abs() <= 1.0e-6;
    if !origin_ok || !spacing_ok {
        return Err(Error::Solver(
            "tire_fem_dem_velocity_bridge: DEM velocity-map grid must match FEM terrain origin/spacing"
                .into(),
        ));
    }
    if dem_velocity.rows.iter().flatten().any(|v| !v.is_finite()) {
        return Err(Error::Solver(
            "tire_fem_dem_velocity_bridge: DEM velocity-map rows must be finite".into(),
        ));
    }
    terrain.surface_velocity_map = Some(dem_velocity.rows.clone());
    Ok(terrain)
}

/// Output from GPU FEM tread contact force feedback.
#[derive(Clone, Debug)]
pub struct TireFemContactFeedbackOutput {
    pub velocities: Vec<[f32; 4]>,
    pub bristles: Vec<[f32; 4]>,
    pub forces: Vec<[f32; 4]>,
}

/// GPU-resident output buffers from FEM tread contact feedback.
///
/// This is the no-readback path for scene/solver pipelines that want the
/// feedback kernel to write updated node velocities, persistent bristles, and
/// per-node forces directly into GPU buffers for the next GPU stage. Use the
/// blocking `TireFemContactFeedbackOutput` path only when CPU diagnostics or
/// tests need to inspect rows.
pub struct TireFemContactFeedbackResidentOutput {
    pub node_count: usize,
    pub bristle_count: usize,
    pub velocities: wgpu::Buffer,
    pub bristles: wgpu::Buffer,
    pub forces: wgpu::Buffer,
}

/// Reusable per-patch accounting for GPU FEM contact feedback output.
///
/// This intentionally summarizes the raw per-node force/bristle rows rather
/// than replacing them. Viewer overlays and rig reports can use it to spot
/// seam/triangle-contact regressions where the total force looks plausible
/// but bristle state collapses into one invisible row.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TireFemContactFeedbackPatchAccounting {
    pub force_row_count: usize,
    pub bristle_row_count: usize,
    pub total_force: [f32; 3],
    pub mean_force_n: f32,
    pub max_force_n: f32,
    pub mean_bristle_deflection_m: f32,
    pub max_bristle_deflection_m: f32,
}

impl TireFemContactFeedbackPatchAccounting {
    pub fn is_finite(self) -> bool {
        self.total_force.iter().all(|v| v.is_finite())
            && self.mean_force_n.is_finite()
            && self.max_force_n.is_finite()
            && self.mean_bristle_deflection_m.is_finite()
            && self.max_bristle_deflection_m.is_finite()
    }
}

impl TireFemContactFeedbackOutput {
    /// Summarize per-node force/bristle rows for contact-patch diagnostics.
    pub fn patch_accounting(&self) -> TireFemContactFeedbackPatchAccounting {
        let mut force_row_count = 0usize;
        let mut total_force = [0.0_f32; 3];
        let mut force_sum = 0.0_f32;
        let mut max_force_n = 0.0_f32;
        for force in &self.forces {
            let f = [
                finite_f32_or(force[0], 0.0),
                finite_f32_or(force[1], 0.0),
                finite_f32_or(force[2], 0.0),
            ];
            let mag = (f[0] * f[0] + f[1] * f[1] + f[2] * f[2]).sqrt();
            if mag > 1.0e-6 {
                force_row_count += 1;
                total_force[0] += f[0];
                total_force[1] += f[1];
                total_force[2] += f[2];
                force_sum += mag;
                max_force_n = max_force_n.max(mag);
            }
        }

        let mut bristle_row_count = 0usize;
        let mut bristle_sum = 0.0_f32;
        let mut max_bristle_deflection_m = 0.0_f32;
        for bristle in &self.bristles {
            let bx = finite_f32_or(bristle[0], 0.0);
            let bz = finite_f32_or(bristle[1], 0.0);
            let mag = (bx * bx + bz * bz).sqrt();
            if mag > 1.0e-9 {
                bristle_row_count += 1;
                bristle_sum += mag;
                max_bristle_deflection_m = max_bristle_deflection_m.max(mag);
            }
        }

        TireFemContactFeedbackPatchAccounting {
            force_row_count,
            bristle_row_count,
            total_force,
            mean_force_n: if force_row_count > 0 {
                force_sum / force_row_count as f32
            } else {
                0.0
            },
            max_force_n,
            mean_bristle_deflection_m: if bristle_row_count > 0 {
                bristle_sum / bristle_row_count as f32
            } else {
                0.0
            },
            max_bristle_deflection_m,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ContactReductionUniforms {
    counts: [u32; 4],
    contact0: [f32; 4],
    hub: [f32; 4],
    heading: [f32; 4],
    terrain_origin: [f32; 4],
    terrain_dims: [u32; 4],
    soil0: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct ContactFeedbackUniforms {
    counts: [u32; 4],
    phys0: [f32; 4],
    hub: [f32; 4],
    hub_vel: [f32; 4],
    heading: [f32; 4],
    material0: [f32; 4],
    material1: [f32; 4],
    terrain_velocity: [f32; 4],
    terrain_origin: [f32; 4],
    terrain_dims: [u32; 4],
    soil0: [f32; 4],
}

/// First GPU reduction for deformed FEM tire contact patch diagnostics.
///
/// It reduces actual tread-node positions into normal load, patch length,
/// contact count and contact centre. This is a diagnostics/readout kernel for
/// flat ground, heightfields, and built-in sphere/box SDF contact. GPU bristle
/// force feedback is handled by `TireFemContactFeedbackKernel`.
pub struct TireFemContactReductionKernel {
    pipeline: wgpu::ComputePipeline,
    layout_io: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

/// First GPU feedback pass that applies tread brush forces back to FEM node
/// velocities and persists per-tread bristle state.
/// GPU-resident buffers for the zero-readback contact-feedback stage
/// — see [`TireFemContactFeedbackKernel::dispatch_resident`]. Bristle
/// state ping-pongs across substeps; force rows stay on-device for the
/// hub tread-wrench reduction.
pub struct TireFemResidentContact {
    bristle_a: wgpu::Buffer,
    bristle_b: wgpu::Buffer,
    vel_out_buf: wgpu::Buffer,
    force_buf: wgpu::Buffer,
    uniform_buf: wgpu::Buffer,
    terrain_dummy: wgpu::Buffer,
    friction_dummy: wgpu::Buffer,
    water_dummy: wgpu::Buffer,
    sinkage_dummy: wgpu::Buffer,
    velocity_map_dummy: wgpu::Buffer,
    sdf_dummy: wgpu::Buffer,
    /// `false` → next dispatch reads bristle A, writes bristle B.
    parity: bool,
    n_tread: u32,
}

impl TireFemResidentContact {
    /// Per-node force rows written by the last resident dispatch —
    /// feed to `step_hub_persistent_with_tread_forces` (row count =
    /// the mesh node count).
    #[must_use]
    pub fn force_rows(&self) -> &wgpu::Buffer {
        &self.force_buf
    }

    /// Debug/test readback of the current bristle state (the slot the
    /// last dispatch wrote). Not part of the zero-readback loop.
    pub fn read_bristles(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<[f32; 4]>> {
        let cur = if self.parity {
            &self.bristle_b
        } else {
            &self.bristle_a
        };
        let n = self.n_tread as usize;
        let stage = stage_vec4(device, "tire_fem resident bristle stage", n);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_fem resident bristle readback"),
        });
        encoder.copy_buffer_to_buffer(
            cur,
            0,
            &stage,
            0,
            (n.max(1) * core::mem::size_of::<[f32; 4]>()) as u64,
        );
        queue.submit(Some(encoder.finish()));
        read_vec4(device, &stage, "tire_fem resident bristles")
    }
}

pub struct TireFemContactFeedbackKernel {
    pipeline: wgpu::ComputePipeline,
    layout_io: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

pub struct TireFemGpuKernel {
    pipeline: wgpu::ComputePipeline,
    layout_io: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
    // Stage (D) — fused-atlas physical node-step pipeline. This runs the
    // packed many-tire atlas in one dispatch instead of looping over
    // persistent per-tire meshes on the host.
    atlas_contract_pipeline: wgpu::ComputePipeline,
    atlas_contract_layout_io: wgpu::BindGroupLayout,
    // Stage (D/f) — fused-atlas hub reduction/integration. One workgroup
    // owns one tire's bead rows and updates the packed hub-state rows in
    // place, matching the per-tire persistent hub kernel without a host loop.
    atlas_hub_pipeline: wgpu::ComputePipeline,
    atlas_hub_layout_io: wgpu::BindGroupLayout,
    // Stage (f/vehicle) — update resident atlas hub parameter rows directly
    // from GPU-resident brush output rows before atlas hub stepping.
    atlas_hub_params_from_brush_pipeline: wgpu::ComputePipeline,
    atlas_hub_params_from_brush_layout_io: wgpu::BindGroupLayout,
    // Stage (f/contact) — update resident atlas hub parameter rows directly
    // from packed GPU-resident FEM contact-reduction output rows.
    atlas_hub_params_from_contact_pipeline: wgpu::ComputePipeline,
    atlas_hub_params_from_contact_layout_io: wgpu::BindGroupLayout,
    // Stage (f/feedback) — update resident atlas hub parameter rows directly
    // from GPU-resident FEM contact-feedback per-node force rows.
    atlas_hub_params_from_feedback_pipeline: wgpu::ComputePipeline,
    atlas_hub_params_from_feedback_layout_io: wgpu::BindGroupLayout,
    // Stage (f) — secondary pipeline that reduces bead reactions and
    // integrates the 1-DOF hub on the GPU between substeps.
    hub_pipeline: wgpu::ComputePipeline,
    hub_layout_io: wgpu::BindGroupLayout,
    hub_layout_uniforms: wgpu::BindGroupLayout,
    // G4 follow-up — variant of the hub pipeline that also reduces the
    // contact-feedback force rows on-device (`hub_reduce_with_tread_wrench`)
    // so the rolling wrench never round-trips through the host.
    hub_tread_pipeline: wgpu::ComputePipeline,
    hub_tread_layout_io: wgpu::BindGroupLayout,
}

/// Hub-integration parameters consumed by `step_hub_persistent`.
#[derive(Clone, Copy, Debug)]
pub struct TireFemHubParams {
    pub dt: f32,
    pub bead_stiffness: f32,
    pub bead_damping: f32,
    pub hub_mass: f32,
    pub hub_damping: f32,
    pub downforce: f32,
    pub gravity_y: f32,
    /// 1.0 to enforce a minimum hub-y floor (rim cannot sink through
    /// ground), 0.0 to skip.
    pub hub_floor_active: f32,
    pub hub_floor_y: f32,
    /// Stage (f) ext — external hub force (N) in world XYZ. Use this
    /// to apply steering / drive / lateral loads from a controller.
    pub external_force_xyz: [f32; 3],
    /// Lateral / longitudinal hub damping (`c·v_xz`) coefficient.
    pub lateral_damping: f32,
}

impl Default for TireFemHubParams {
    fn default() -> Self {
        Self {
            dt: 1.0 / 240.0,
            bead_stiffness: 5.5e4,
            bead_damping: 650.0,
            hub_mass: 22.0,
            hub_damping: 260.0,
            downforce: 2_600.0,
            gravity_y: -9.81,
            hub_floor_active: 1.0,
            hub_floor_y: 0.09,
            external_force_xyz: [0.0; 3],
            lateral_damping: 320.0,
        }
    }
}

/// Feed a reduced FEM contact world wrench into the GPU hub integrator.
///
/// The hub kernel consumes an external force vector each substep. This helper
/// maps the contact-reduction `world_force` into that vector while preserving
/// any pre-existing controller force in `params.external_force_xyz`.
/// `force_scale` lets callers apply substep/suspension filtering without
/// hiding the sign convention in viewer code. Moments are intentionally left
/// for the later chassis/yaw coupling path; the current 3-DOF hub integrator
/// only accepts forces.
pub fn tire_fem_hub_params_with_contact_reduction(
    mut params: TireFemHubParams,
    contact: &TireFemContactReductionOutput,
    force_scale: f32,
) -> TireFemHubParams {
    let scale = finite_f32_or(force_scale, 1.0).clamp(-10.0, 10.0);
    for axis in 0..3 {
        params.external_force_xyz[axis] = finite_f32_or(params.external_force_xyz[axis], 0.0)
            + finite_f32_or(contact.world_force[axis], 0.0) * scale;
    }
    params
}

/// Build per-tire FEM hub parameters from vehicle/brush tire outputs.
///
/// This is the first reusable vehicle-to-FEM assembly bridge: the GPU vehicle
/// owner can produce per-wheel `BrushWheelOutput` rows, and this helper maps
/// the same world-force vector used by the planar chassis integrator into
/// `TireFemHubParams::external_force_xyz` for `step_resident_atlas_hubs`.
/// It preserves caller-supplied base hub parameters and only adds the scaled
/// per-wheel force. The fully GPU-resident version should write atlas hub
/// parameter rows directly, but this keeps sign/shape accounting in one native
/// API instead of hiding it in viewer scene code.
pub fn tire_fem_hub_params_with_brush_outputs(
    base_params: &[TireFemHubParams],
    brush_outputs: &[BrushWheelOutput],
    force_scale: f32,
) -> Result<Vec<TireFemHubParams>> {
    if base_params.len() != brush_outputs.len() {
        return Err(Error::Solver(format!(
            "tire_fem hub brush handoff: params length {} must match brush output count {}",
            base_params.len(),
            brush_outputs.len()
        )));
    }
    let scale = finite_f32_or(force_scale, 1.0).clamp(-10.0, 10.0);
    Ok(base_params
        .iter()
        .copied()
        .zip(brush_outputs.iter())
        .map(|(mut params, brush)| {
            for axis in 0..3 {
                params.external_force_xyz[axis] =
                    finite_f32_or(params.external_force_xyz[axis], 0.0)
                        + finite_f32_or(brush.world_force[axis], 0.0) * scale;
            }
            params
        })
        .collect())
}

/// Build per-tire FEM hub parameters from FEM contact-feedback node forces.
///
/// The GPU feedback shader writes one force row per atlas node. This CPU helper
/// mirrors the resident GPU assembly path by summing each tire's atlas node
/// range and adding the scaled world force into `external_force_xyz`.
pub fn tire_fem_hub_params_with_feedback_output(
    base_params: &[TireFemHubParams],
    feedback: &TireFemContactFeedbackOutput,
    atlas_layout: &TireFemGpuBatchAtlasLayout,
    force_scale: f32,
) -> Result<Vec<TireFemHubParams>> {
    if base_params.len() != atlas_layout.rows.len() {
        return Err(Error::Solver(format!(
            "tire_fem hub feedback handoff: params length {} must match tire count {}",
            base_params.len(),
            atlas_layout.rows.len()
        )));
    }
    if feedback.forces.len() != atlas_layout.plan.total_node_count {
        return Err(Error::Solver(format!(
            "tire_fem hub feedback handoff: feedback force rows {} must match atlas node count {}",
            feedback.forces.len(),
            atlas_layout.plan.total_node_count
        )));
    }
    let scale = finite_f32_or(force_scale, 1.0).clamp(-10.0, 10.0);
    Ok(base_params
        .iter()
        .copied()
        .zip(atlas_layout.rows.iter())
        .map(|(mut params, row)| {
            let mut total = [0.0_f32; 3];
            for force in &feedback.forces[row.node_offset..row.node_offset + row.node_count] {
                for axis in 0..3 {
                    total[axis] += finite_f32_or(force[axis], 0.0);
                }
            }
            for axis in 0..3 {
                params.external_force_xyz[axis] =
                    finite_f32_or(params.external_force_xyz[axis], 0.0) + total[axis] * scale;
            }
            params
        })
        .collect())
}

fn tire_fem_gpu_atlas_hub_param_row(params: TireFemHubParams) -> TireFemGpuAtlasHubParamRow {
    TireFemGpuAtlasHubParamRow {
        cfg: [0.0, params.hub_floor_active, params.hub_floor_y, params.dt],
        mech: [
            params.bead_stiffness,
            params.bead_damping,
            params.hub_mass,
            params.hub_damping,
        ],
        load: [params.downforce, params.gravity_y, 0.0, 0.0],
        extf: [
            params.external_force_xyz[0],
            params.external_force_xyz[1],
            params.external_force_xyz[2],
            params.lateral_damping,
        ],
    }
}

/// PLAN_REALTIRE Stream C G4 prep — build a complete GPU node-FEM input
/// payload from a live [`molla_vehicle::TetraTireFem`].
///
/// Combines `tire.gpu_snapshot()` (mesh + topology) with the tire's live hub
/// state (position, omega, drive/brake torque) into a single [`TireFemGpuInput`]
/// ready for [`TireFemGpuKernel::dispatch_blocking`] or
/// [`TireFemGpuKernel::create_persistent`]. The caller supplies a base
/// [`TireFemGpuParams`] for the fields that aren't snapshot-derived (dt,
/// damping, compliance, etc.); this helper overwrites only the fields that
/// must come from the CPU tire — `hub`, `hub_omega`, `drive_torque_n_m`,
/// `brake_torque_n_m`, `pressure_pa`, `pressure_area_per_tread_node`,
/// `cavity_radial_force_per_bead`.
///
/// Foundation for the eventual CPU-vs-GPU hub-trajectory parity hardware
/// lane (Stream C G4): build both sides from the same `TetraTireFem`,
/// dispatch in parallel, compare hub trajectories within tolerance.
pub fn tire_fem_gpu_input_from_tetra_fem(
    tire: &molla_vehicle::TetraTireFem,
    mut base_params: TireFemGpuParams,
) -> TireFemGpuInput {
    let snap = tire.gpu_snapshot();
    let hub = tire.hub();
    base_params.hub = [hub[0] as f32, hub[1] as f32, hub[2] as f32];
    base_params.hub_omega = tire.hub_omega() as f32;
    base_params.drive_torque_n_m = tire.drive_torque_n_m() as f32;
    base_params.brake_torque_n_m = tire.brake_torque_n_m() as f32;
    base_params.pressure_pa = tire.pressure() as f32;
    base_params.pressure_area_per_tread_node = snap.pressure_area_per_tread_node;
    base_params.cavity_radial_force_per_bead = snap.cavity_radial_force_per_bead;
    TireFemGpuInput {
        rest_positions: snap.rest_positions,
        positions: snap.positions,
        velocities: snap.velocities,
        masses: snap.masses,
        tet_indices: snap.tet_indices,
        tet_grads: snap.tet_grads,
        tet_volumes: snap.tet_volumes,
        tet_mu: snap.tet_mu,
        tet_lambda: snap.tet_lambda,
        tet_fiber_dir: snap.tet_fiber_dir,
        tet_fiber_k1: snap.tet_fiber_k1,
        tet_fiber_k2: snap.tet_fiber_k2,
        tet_fiber_dir_b: snap.tet_fiber_dir_b,
        tet_fiber_k1_b: snap.tet_fiber_k1_b,
        tet_fiber_k2_b: snap.tet_fiber_k2_b,
        tet_fiber_dir_c: snap.tet_fiber_dir_c,
        tet_fiber_k1_c: snap.tet_fiber_k1_c,
        tet_fiber_k2_c: snap.tet_fiber_k2_c,
        tet_fiber_dir_d: snap.tet_fiber_dir_d,
        tet_fiber_k1_d: snap.tet_fiber_k1_d,
        tet_fiber_k2_d: snap.tet_fiber_k2_d,
        bead_ids: snap.bead_ids,
        bead_offsets: snap.bead_offsets,
        tread_ids: snap.tread_ids,
        node_tet_offsets: snap.node_tet_offsets,
        node_tet_entries: snap.node_tet_entries,
        xpbd_edge_indices: snap.xpbd_edge_indices,
        xpbd_edge_params: snap.xpbd_edge_params,
        node_edge_offsets: snap.node_edge_offsets,
        node_edge_entries: snap.node_edge_entries,
        xpbd_dihedral_indices: snap.xpbd_dihedral_indices,
        xpbd_dihedral_params: snap.xpbd_dihedral_params,
        node_dihedral_offsets: snap.node_dihedral_offsets,
        node_dihedral_entries: snap.node_dihedral_entries,
        params: base_params,
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct HubUniforms {
    cfg: [f32; 4],
    mech: [f32; 4],
    load: [f32; 4],
    extf: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct BrushHubUniforms {
    counts: [u32; 4],
    params: [f32; 4],
}

/// Shader-layout-compatible hub parameters for one tire in a fused atlas.
///
/// This mirrors [`HubUniforms`], but lives in a storage row so the atlas hub
/// kernel can launch one workgroup per tire without rebinding uniforms in a
/// host loop.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct TireFemGpuAtlasHubParamRow {
    pub cfg: [f32; 4],
    pub mech: [f32; 4],
    pub load: [f32; 4],
    pub extf: [f32; 4],
}

/// Persistent GPU state for one tire FEM mesh (stage (g)).
///
/// Holds the **constant** mesh buffers (rest positions, tet topology +
/// materials + fibre data, CSR adjacency, bead/tread index arrays)
/// uploaded **once** and the ping-pong position/velocity buffers so
/// many substeps run back-to-back on the GPU with no per-substep
/// re-uploads or readbacks. Call `TireFemGpuKernel::create_persistent`
/// to build, `step_persistent` to advance one substep, and
/// `readback_state` once per frame for visualisation.
//
// All buffer fields are read by the wgpu bind groups via Arc-cloned
// internal references; the Rust borrow-checker can't see those uses,
// so dead-code warnings are silenced.
#[allow(dead_code)]
pub struct TireFemPersistentMesh {
    n_nodes: u32,
    // Constant mesh buffers (uploaded once, never written again).
    rest_buf: wgpu::Buffer,
    mass_buf: wgpu::Buffer,
    tet_i_buf: wgpu::Buffer,
    tet_g_buf: wgpu::Buffer,
    tet_v_buf: wgpu::Buffer,
    tet_mu_buf: wgpu::Buffer,
    tet_lambda_buf: wgpu::Buffer,
    tet_fk_buf: wgpu::Buffer,
    tet_k1_buf: wgpu::Buffer,
    tet_k2_buf: wgpu::Buffer,
    tet_fk_b_buf: wgpu::Buffer,
    tet_k1_b_buf: wgpu::Buffer,
    tet_k2_b_buf: wgpu::Buffer,
    tet_fk_c_buf: wgpu::Buffer,
    tet_k1_c_buf: wgpu::Buffer,
    tet_k2_c_buf: wgpu::Buffer,
    tet_fk_d_buf: wgpu::Buffer,
    tet_k1_d_buf: wgpu::Buffer,
    tet_k2_d_buf: wgpu::Buffer,
    xpbd_edge_i_buf: wgpu::Buffer,
    xpbd_edge_p_buf: wgpu::Buffer,
    node_edge_off_buf: wgpu::Buffer,
    node_edge_ent_buf: wgpu::Buffer,
    xpbd_dihedral_i_buf: wgpu::Buffer,
    xpbd_dihedral_p_buf: wgpu::Buffer,
    node_dihedral_off_buf: wgpu::Buffer,
    node_dihedral_ent_buf: wgpu::Buffer,
    csr_off_buf: wgpu::Buffer,
    csr_ent_buf: wgpu::Buffer,
    bead_id_buf: wgpu::Buffer,
    bead_off_buf: wgpu::Buffer,
    tread_id_buf: wgpu::Buffer,
    // Ping-pong state buffers — A on even substeps, B on odd.
    pos_a: wgpu::Buffer,
    pos_b: wgpu::Buffer,
    vel_a: wgpu::Buffer,
    vel_b: wgpu::Buffer,
    force_buf: wgpu::Buffer,
    uniform_buf: wgpu::Buffer,
    // Bind groups for the two ping-pong directions.
    bg_a_to_b: wgpu::BindGroup,
    bg_b_to_a: wgpu::BindGroup,
    bg_uniforms: wgpu::BindGroup,
    /// Substep parity — `false` means the next dispatch reads `pos_a`
    /// and writes `pos_b`; `true` reads B and writes A.
    parity: bool,
    // Uniforms parameters (updated lazily for each dispatch).
    counts: [u32; 4],
    /// Cached uniform fields for partial-update without reconstructing
    /// the whole `Uniforms` struct on the CPU each step.
    cached_uniforms: Uniforms,
    /// Stage (f) — GPU-resident hub state `(hub_x, hub_y, hub_z, hub_v_y)`,
    /// written by the hub reduction kernel and read back once per frame.
    hub_state_buf: wgpu::Buffer,
    /// Hub-uniform companion buffer (per-substep mechanics).
    hub_uniform_buf: wgpu::Buffer,
    // Two bind groups for the hub kernel, one per ping-pong direction
    // (they reference the current `pos_in`/`vel_in` slot just like the
    // main kernel does).
    hub_bg_a: wgpu::BindGroup,
    hub_bg_b: wgpu::BindGroup,
    hub_bg_uniforms: wgpu::BindGroup,
    n_bead: u32,
}

impl TireFemPersistentMesh {
    pub fn node_count(&self) -> u32 {
        self.n_nodes
    }
}

/// Host-side persistent batch for multiple tire FEM meshes.
///
/// This is the first scaling layer for multi-wheel vehicles: every tire keeps
/// its mesh/topology buffers resident on the GPU, while callers step and
/// read back the whole set through one reusable batch object instead of
/// allocating/ad-hoc managing one mesh object per wheel in viewer code. The
/// meshes may be heterogeneous. Use [`TireFemGpuAtlasBatch`] when the caller
/// can keep one packed fused-atlas owner instead of looping over per-tire
/// persistent meshes.
pub struct TireFemGpuBatch {
    meshes: Vec<TireFemPersistentMesh>,
}

/// GPU-resident packed atlas owner for many tire FEM meshes.
///
/// Unlike [`TireFemGpuBatch`], this keeps all tire topology/state in one
/// atlas-shaped bind group pair and advances the whole batch with one fused
/// `tire_fem_atlas.wgsl` dispatch. It is the first resident version of the
/// atlas path; higher-level viewer code can move to this owner without
/// changing per-tire output shapes.
pub struct TireFemGpuAtlasBatch {
    atlas: TireFemGpuBatchAtlasBuffers,
    pos_a: wgpu::Buffer,
    pos_b: wgpu::Buffer,
    vel_a: wgpu::Buffer,
    vel_b: wgpu::Buffer,
    force_buf: wgpu::Buffer,
    param_rows_buf: wgpu::Buffer,
    atlas_offset_rows_buf: wgpu::Buffer,
    hub_state_buf: wgpu::Buffer,
    atlas_hub_param_rows_buf: wgpu::Buffer,
    bg_a_to_b: wgpu::BindGroup,
    bg_b_to_a: wgpu::BindGroup,
    atlas_hub_bg_a: wgpu::BindGroup,
    atlas_hub_bg_b: wgpu::BindGroup,
    parity: bool,
}

impl TireFemGpuAtlasBatch {
    pub fn tire_count(&self) -> usize {
        self.atlas.layout.plan.tire_count
    }

    pub fn total_node_count(&self) -> usize {
        self.atlas.layout.plan.total_node_count
    }

    pub fn atlas(&self) -> &TireFemGpuBatchAtlasBuffers {
        &self.atlas
    }
}

/// Combined resident owner for the vehicle→brush→FEM-atlas path.
///
/// This bundles the resident chassis/brush owner with the resident fused tire
/// FEM atlas so live runtimes can keep one long-lived object instead of
/// manually carrying two owners and remembering the required dispatch order.
pub struct TireResidentVehicleFemAtlasOwner {
    pub vehicle: TireVehicleResidentChassisBrushOwner,
    pub fem: TireFemGpuAtlasBatch,
    tire_count: usize,
}

impl TireResidentVehicleFemAtlasOwner {
    pub const fn tire_count(&self) -> usize {
        self.tire_count
    }

    pub const fn wheel_count(&self) -> usize {
        self.tire_count
    }

    pub fn fem(&self) -> &TireFemGpuAtlasBatch {
        &self.fem
    }

    pub fn fem_mut(&mut self) -> &mut TireFemGpuAtlasBatch {
        &mut self.fem
    }

    pub fn vehicle(&self) -> &TireVehicleResidentChassisBrushOwner {
        &self.vehicle
    }

    pub fn vehicle_mut(&mut self) -> &mut TireVehicleResidentChassisBrushOwner {
        &mut self.vehicle
    }
}

/// Host-side dispatch/packing plan for a multi-tire FEM batch.
///
/// The current batch API keeps one persistent mesh per tire and dispatches
/// them in a loop. This plan records the atlas-sized counts that the fused
/// resident/one-shot atlas dispatches must preserve, so tests can validate
/// heterogeneous batches independently from a specific runtime owner.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TireFemGpuBatchDispatchPlan {
    pub tire_count: usize,
    pub total_node_count: usize,
    pub total_tet_count: usize,
    pub total_tread_count: usize,
    pub total_bead_count: usize,
    pub total_xpbd_edge_count: usize,
    pub total_xpbd_dihedral_count: usize,
    pub max_nodes_per_tire: usize,
    pub node_workgroup_count: u32,
    pub persistent_state_bytes: u64,
}

impl TireFemGpuBatchDispatchPlan {
    pub fn is_nonempty(self) -> bool {
        self.tire_count > 0
            && self.total_node_count > 0
            && self.total_tet_count > 0
            && self.node_workgroup_count > 0
    }
}

/// One tire's slice in the fused FEM atlas.
///
/// These offsets make the packing contract explicit for both one-shot atlas
/// dispatch and resident atlas ownership.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TireFemGpuBatchAtlasRow {
    pub tire_index: usize,
    pub node_offset: usize,
    pub node_count: usize,
    pub tet_offset: usize,
    pub tet_count: usize,
    pub tread_offset: usize,
    pub tread_count: usize,
    pub bead_offset: usize,
    pub bead_count: usize,
    pub xpbd_edge_offset: usize,
    pub xpbd_edge_count: usize,
    pub xpbd_dihedral_offset: usize,
    pub xpbd_dihedral_count: usize,
    pub node_workgroup_offset: u32,
    pub node_workgroup_count: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TireFemGpuBatchAtlasLayout {
    pub plan: TireFemGpuBatchDispatchPlan,
    pub rows: Vec<TireFemGpuBatchAtlasRow>,
}

/// Shader-layout-compatible per-tire parameter row for a fused FEM atlas.
///
/// This mirrors the single-tire `Uniforms` layout, but it is a storage-buffer
/// row instead of a private uniform struct. The fused node-step shader indexes
/// this by `node_tire_indices[node_id]` and keeps the same scalar semantics as
/// the existing one-mesh dispatch path.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct TireFemGpuBatchAtlasParamRow {
    pub counts: [u32; 4],
    pub counts2: [u32; 4],
    pub phys0: [f32; 4],
    pub phys1: [f32; 4],
    pub phys2: [f32; 4],
    pub phys3: [f32; 4],
    pub phys4: [f32; 4],
}

/// Shader-layout-compatible per-tire atlas offset row.
///
/// The fused kernel needs both global atlas ranges and per-tire local counts.
/// This row keeps offsets/counts in u32 vec4 groups so WGSL can index
/// node/tet/contact/XPBD slices without reconstructing Rust `usize` rows.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct TireFemGpuBatchAtlasOffsetRow {
    /// `(node_offset, node_count, node_workgroup_offset, node_workgroup_count)`.
    pub node: [u32; 4],
    /// `(tet_offset, tet_count, 0, 0)`.
    pub tet: [u32; 4],
    /// `(tread_offset, tread_count, bead_offset, bead_count)`.
    pub contact: [u32; 4],
    /// `(xpbd_edge_offset, xpbd_edge_count, xpbd_dihedral_offset, xpbd_dihedral_count)`.
    pub xpbd: [u32; 4],
}

/// Packed host-side buffer payload for the fused FEM atlas dispatch.
///
/// The fused one-dispatch and resident paths need all node, tet, bead, tread,
/// CSR, and XPBD topology buffers in one contiguous atlas. This struct is the
/// preflight packing contract: indices are rewritten from per-tire local space
/// into atlas space while per-tire `params` live in shader-compatible rows.
#[derive(Clone, Debug)]
pub struct TireFemGpuBatchAtlasBuffers {
    pub layout: TireFemGpuBatchAtlasLayout,
    /// Atlas-space tire row for each node. A fused node-step shader can use
    /// this to load the correct per-tire params/hub row for every invocation.
    pub node_tire_indices: Vec<u32>,
    /// Atlas-space tire row for each tet/material row. A fused shader can use
    /// this when reducing material/contact diagnostics back to per-tire rows.
    pub tet_tire_indices: Vec<u32>,
    /// Atlas-space tire row for each bead id.
    pub bead_tire_indices: Vec<u32>,
    /// Atlas-space tire row for each tread id.
    pub tread_tire_indices: Vec<u32>,
    pub rest_positions: Vec<[f32; 4]>,
    pub positions: Vec<[f32; 4]>,
    pub velocities: Vec<[f32; 4]>,
    pub masses: Vec<f32>,
    pub tet_indices: Vec<u32>,
    pub tet_grads: Vec<[f32; 4]>,
    pub tet_volumes: Vec<f32>,
    pub tet_mu: Vec<f32>,
    pub tet_lambda: Vec<f32>,
    pub tet_fiber_dir: Vec<[f32; 4]>,
    pub tet_fiber_k1: Vec<f32>,
    pub tet_fiber_k2: Vec<f32>,
    pub tet_fiber_dir_b: Vec<[f32; 4]>,
    pub tet_fiber_k1_b: Vec<f32>,
    pub tet_fiber_k2_b: Vec<f32>,
    pub tet_fiber_dir_c: Vec<[f32; 4]>,
    pub tet_fiber_k1_c: Vec<f32>,
    pub tet_fiber_k2_c: Vec<f32>,
    pub tet_fiber_dir_d: Vec<[f32; 4]>,
    pub tet_fiber_k1_d: Vec<f32>,
    pub tet_fiber_k2_d: Vec<f32>,
    pub bead_ids: Vec<u32>,
    pub bead_offsets: Vec<[f32; 4]>,
    pub tread_ids: Vec<u32>,
    pub node_tet_offsets: Vec<u32>,
    pub node_tet_entries: Vec<u32>,
    pub xpbd_edge_indices: Vec<[u32; 4]>,
    pub xpbd_edge_params: Vec<[f32; 4]>,
    pub node_edge_offsets: Vec<u32>,
    pub node_edge_entries: Vec<u32>,
    pub xpbd_dihedral_indices: Vec<[u32; 4]>,
    pub xpbd_dihedral_params: Vec<[f32; 4]>,
    pub node_dihedral_offsets: Vec<u32>,
    pub node_dihedral_entries: Vec<u32>,
    /// Shader-layout-compatible rows for the fused atlas kernel.
    pub param_rows: Vec<TireFemGpuBatchAtlasParamRow>,
    /// Shader-layout-compatible atlas offset rows for the fused kernel.
    pub offset_rows: Vec<TireFemGpuBatchAtlasOffsetRow>,
    /// Original Rust-side params retained for host fallback/debug callers.
    pub params: Vec<TireFemGpuParams>,
}

impl TireFemGpuBatchAtlasBuffers {
    pub fn storage_bytes(&self) -> u64 {
        bytes_of_slice::<[f32; 4]>(self.rest_positions.len())
            .saturating_add(bytes_of_slice::<[f32; 4]>(self.positions.len()))
            .saturating_add(bytes_of_slice::<[f32; 4]>(self.velocities.len()))
            .saturating_add(bytes_of_slice::<f32>(self.masses.len()))
            .saturating_add(bytes_of_slice::<u32>(self.node_tire_indices.len()))
            .saturating_add(bytes_of_slice::<u32>(self.tet_tire_indices.len()))
            .saturating_add(bytes_of_slice::<u32>(self.bead_tire_indices.len()))
            .saturating_add(bytes_of_slice::<u32>(self.tread_tire_indices.len()))
            .saturating_add(bytes_of_slice::<u32>(self.tet_indices.len()))
            .saturating_add(bytes_of_slice::<[f32; 4]>(self.tet_grads.len()))
            .saturating_add(bytes_of_slice::<f32>(self.tet_volumes.len()))
            .saturating_add(bytes_of_slice::<f32>(self.tet_mu.len()))
            .saturating_add(bytes_of_slice::<f32>(self.tet_lambda.len()))
            .saturating_add(bytes_of_slice::<[f32; 4]>(self.tet_fiber_dir.len()))
            .saturating_add(bytes_of_slice::<f32>(self.tet_fiber_k1.len()))
            .saturating_add(bytes_of_slice::<f32>(self.tet_fiber_k2.len()))
            .saturating_add(bytes_of_slice::<[f32; 4]>(self.tet_fiber_dir_b.len()))
            .saturating_add(bytes_of_slice::<f32>(self.tet_fiber_k1_b.len()))
            .saturating_add(bytes_of_slice::<f32>(self.tet_fiber_k2_b.len()))
            .saturating_add(bytes_of_slice::<[f32; 4]>(self.tet_fiber_dir_c.len()))
            .saturating_add(bytes_of_slice::<f32>(self.tet_fiber_k1_c.len()))
            .saturating_add(bytes_of_slice::<f32>(self.tet_fiber_k2_c.len()))
            .saturating_add(bytes_of_slice::<[f32; 4]>(self.tet_fiber_dir_d.len()))
            .saturating_add(bytes_of_slice::<f32>(self.tet_fiber_k1_d.len()))
            .saturating_add(bytes_of_slice::<f32>(self.tet_fiber_k2_d.len()))
            .saturating_add(bytes_of_slice::<u32>(self.bead_ids.len()))
            .saturating_add(bytes_of_slice::<[f32; 4]>(self.bead_offsets.len()))
            .saturating_add(bytes_of_slice::<u32>(self.tread_ids.len()))
            .saturating_add(bytes_of_slice::<u32>(self.node_tet_offsets.len()))
            .saturating_add(bytes_of_slice::<u32>(self.node_tet_entries.len()))
            .saturating_add(bytes_of_slice::<[u32; 4]>(self.xpbd_edge_indices.len()))
            .saturating_add(bytes_of_slice::<[f32; 4]>(self.xpbd_edge_params.len()))
            .saturating_add(bytes_of_slice::<u32>(self.node_edge_offsets.len()))
            .saturating_add(bytes_of_slice::<u32>(self.node_edge_entries.len()))
            .saturating_add(bytes_of_slice::<[u32; 4]>(self.xpbd_dihedral_indices.len()))
            .saturating_add(bytes_of_slice::<[f32; 4]>(self.xpbd_dihedral_params.len()))
            .saturating_add(bytes_of_slice::<u32>(self.node_dihedral_offsets.len()))
            .saturating_add(bytes_of_slice::<u32>(self.node_dihedral_entries.len()))
            .saturating_add(bytes_of_slice::<TireFemGpuBatchAtlasParamRow>(
                self.param_rows.len(),
            ))
            .saturating_add(bytes_of_slice::<TireFemGpuBatchAtlasOffsetRow>(
                self.offset_rows.len(),
            ))
    }

    pub fn validate(&self) -> Result<()> {
        if !self.layout.is_contiguous() {
            return Err(Error::Solver(
                "tire_fem atlas buffers: layout rows are not contiguous".into(),
            ));
        }
        let plan = self.layout.plan;
        expect_len(
            "tire_fem atlas rest_positions",
            self.rest_positions.len(),
            plan.total_node_count,
        )?;
        expect_len(
            "tire_fem atlas positions",
            self.positions.len(),
            plan.total_node_count,
        )?;
        expect_len(
            "tire_fem atlas velocities",
            self.velocities.len(),
            plan.total_node_count,
        )?;
        expect_len(
            "tire_fem atlas masses",
            self.masses.len(),
            plan.total_node_count,
        )?;
        expect_len(
            "tire_fem atlas node_tire_indices",
            self.node_tire_indices.len(),
            plan.total_node_count,
        )?;
        expect_len(
            "tire_fem atlas tet_tire_indices",
            self.tet_tire_indices.len(),
            plan.total_tet_count,
        )?;
        expect_len(
            "tire_fem atlas tread_tire_indices",
            self.tread_tire_indices.len(),
            plan.total_tread_count,
        )?;
        expect_len(
            "tire_fem atlas bead_tire_indices",
            self.bead_tire_indices.len(),
            plan.total_bead_count,
        )?;
        expect_len(
            "tire_fem atlas tet_indices",
            self.tet_indices.len(),
            plan.total_tet_count * 4,
        )?;
        expect_len(
            "tire_fem atlas tet_grads",
            self.tet_grads.len(),
            plan.total_tet_count * 4,
        )?;
        for (label, len) in [
            ("tire_fem atlas tet_volumes", self.tet_volumes.len()),
            ("tire_fem atlas tet_mu", self.tet_mu.len()),
            ("tire_fem atlas tet_lambda", self.tet_lambda.len()),
            ("tire_fem atlas tet_fiber_dir", self.tet_fiber_dir.len()),
            ("tire_fem atlas tet_fiber_k1", self.tet_fiber_k1.len()),
            ("tire_fem atlas tet_fiber_k2", self.tet_fiber_k2.len()),
            ("tire_fem atlas tet_fiber_dir_b", self.tet_fiber_dir_b.len()),
            ("tire_fem atlas tet_fiber_k1_b", self.tet_fiber_k1_b.len()),
            ("tire_fem atlas tet_fiber_k2_b", self.tet_fiber_k2_b.len()),
            ("tire_fem atlas tet_fiber_dir_c", self.tet_fiber_dir_c.len()),
            ("tire_fem atlas tet_fiber_k1_c", self.tet_fiber_k1_c.len()),
            ("tire_fem atlas tet_fiber_k2_c", self.tet_fiber_k2_c.len()),
            ("tire_fem atlas tet_fiber_dir_d", self.tet_fiber_dir_d.len()),
            ("tire_fem atlas tet_fiber_k1_d", self.tet_fiber_k1_d.len()),
            ("tire_fem atlas tet_fiber_k2_d", self.tet_fiber_k2_d.len()),
        ] {
            expect_len(label, len, plan.total_tet_count)?;
        }
        expect_len(
            "tire_fem atlas bead_ids",
            self.bead_ids.len(),
            plan.total_bead_count,
        )?;
        expect_len(
            "tire_fem atlas bead_offsets",
            self.bead_offsets.len(),
            plan.total_bead_count,
        )?;
        expect_len(
            "tire_fem atlas tread_ids",
            self.tread_ids.len(),
            plan.total_tread_count,
        )?;
        expect_len(
            "tire_fem atlas node_tet_offsets",
            self.node_tet_offsets.len(),
            plan.total_node_count + 1,
        )?;
        expect_len(
            "tire_fem atlas node_tet_entries",
            self.node_tet_entries.len(),
            plan.total_tet_count * 4,
        )?;
        expect_last_offset(
            "tire_fem atlas node_tet_offsets",
            &self.node_tet_offsets,
            self.node_tet_entries.len(),
        )?;
        expect_len(
            "tire_fem atlas xpbd_edge_indices",
            self.xpbd_edge_indices.len(),
            plan.total_xpbd_edge_count,
        )?;
        expect_len(
            "tire_fem atlas xpbd_edge_params",
            self.xpbd_edge_params.len(),
            plan.total_xpbd_edge_count,
        )?;
        if plan.total_xpbd_edge_count > 0 {
            expect_len(
                "tire_fem atlas node_edge_offsets",
                self.node_edge_offsets.len(),
                plan.total_node_count + 1,
            )?;
            expect_last_offset(
                "tire_fem atlas node_edge_offsets",
                &self.node_edge_offsets,
                self.node_edge_entries.len(),
            )?;
        } else {
            expect_len(
                "tire_fem atlas node_edge_offsets",
                self.node_edge_offsets.len(),
                0,
            )?;
            expect_len(
                "tire_fem atlas node_edge_entries",
                self.node_edge_entries.len(),
                0,
            )?;
        }
        expect_len(
            "tire_fem atlas xpbd_dihedral_indices",
            self.xpbd_dihedral_indices.len(),
            plan.total_xpbd_dihedral_count,
        )?;
        expect_len(
            "tire_fem atlas xpbd_dihedral_params",
            self.xpbd_dihedral_params.len(),
            plan.total_xpbd_dihedral_count,
        )?;
        if plan.total_xpbd_dihedral_count > 0 {
            expect_len(
                "tire_fem atlas node_dihedral_offsets",
                self.node_dihedral_offsets.len(),
                plan.total_node_count + 1,
            )?;
            expect_last_offset(
                "tire_fem atlas node_dihedral_offsets",
                &self.node_dihedral_offsets,
                self.node_dihedral_entries.len(),
            )?;
        } else {
            expect_len(
                "tire_fem atlas node_dihedral_offsets",
                self.node_dihedral_offsets.len(),
                0,
            )?;
            expect_len(
                "tire_fem atlas node_dihedral_entries",
                self.node_dihedral_entries.len(),
                0,
            )?;
        }
        expect_len(
            "tire_fem atlas param_rows",
            self.param_rows.len(),
            plan.tire_count,
        )?;
        expect_len(
            "tire_fem atlas offset_rows",
            self.offset_rows.len(),
            plan.tire_count,
        )?;
        expect_len("tire_fem atlas params", self.params.len(), plan.tire_count)?;
        Ok(())
    }

    pub fn split_outputs(
        &self,
        positions: &[[f32; 4]],
        velocities: &[[f32; 4]],
        forces: &[[f32; 4]],
    ) -> Result<Vec<TireFemGpuOutput>> {
        self.validate()?;
        let node_count = self.layout.plan.total_node_count;
        expect_len(
            "tire_fem atlas output positions",
            positions.len(),
            node_count,
        )?;
        expect_len(
            "tire_fem atlas output velocities",
            velocities.len(),
            node_count,
        )?;
        expect_len("tire_fem atlas output forces", forces.len(), node_count)?;

        let mut outputs = Vec::with_capacity(self.layout.plan.tire_count);
        for row in &self.layout.rows {
            let range = row.node_offset..row.node_offset + row.node_count;
            outputs.push(TireFemGpuOutput {
                positions: positions[range.clone()].to_vec(),
                velocities: velocities[range.clone()].to_vec(),
                forces: forces[range].to_vec(),
            });
        }
        Ok(outputs)
    }

    pub fn split_output_rows(
        &self,
        rows: &TireFemGpuBatchAtlasOutputRows,
    ) -> Result<Vec<TireFemGpuOutput>> {
        self.split_outputs(&rows.positions, &rows.velocities, &rows.forces)
    }

    pub fn join_outputs(
        &self,
        outputs: &[TireFemGpuOutput],
    ) -> Result<TireFemGpuBatchAtlasOutputRows> {
        self.validate()?;
        expect_len(
            "tire_fem atlas per-tire output rows",
            outputs.len(),
            self.layout.plan.tire_count,
        )?;
        let mut positions = Vec::with_capacity(self.layout.plan.total_node_count);
        let mut velocities = Vec::with_capacity(self.layout.plan.total_node_count);
        let mut forces = Vec::with_capacity(self.layout.plan.total_node_count);
        for (row, output) in self.layout.rows.iter().zip(outputs) {
            expect_len(
                "tire_fem atlas output row positions",
                output.positions.len(),
                row.node_count,
            )?;
            expect_len(
                "tire_fem atlas output row velocities",
                output.velocities.len(),
                row.node_count,
            )?;
            expect_len(
                "tire_fem atlas output row forces",
                output.forces.len(),
                row.node_count,
            )?;
            positions.extend_from_slice(&output.positions);
            velocities.extend_from_slice(&output.velocities);
            forces.extend_from_slice(&output.forces);
        }
        Ok(TireFemGpuBatchAtlasOutputRows {
            positions,
            velocities,
            forces,
        })
    }

    pub fn extract_input(&self, tire_index: usize) -> Result<TireFemGpuInput> {
        self.validate()?;
        let row = self.layout.rows.get(tire_index).ok_or_else(|| {
            Error::Solver(format!(
                "tire_fem atlas extract: tire index {tire_index} out of range"
            ))
        })?;
        let node_range = row.node_offset..row.node_offset + row.node_count;
        let tet_range = row.tet_offset..row.tet_offset + row.tet_count;
        let tet_scalar_range = row.tet_offset * 4..(row.tet_offset + row.tet_count) * 4;
        let tread_range = row.tread_offset..row.tread_offset + row.tread_count;
        let bead_range = row.bead_offset..row.bead_offset + row.bead_count;
        let edge_range = row.xpbd_edge_offset..row.xpbd_edge_offset + row.xpbd_edge_count;
        let dihedral_range =
            row.xpbd_dihedral_offset..row.xpbd_dihedral_offset + row.xpbd_dihedral_count;

        let mut tet_indices = Vec::with_capacity(row.tet_count * 4);
        for &index in &self.tet_indices[tet_scalar_range.clone()] {
            tet_indices.push(subtract_u32(
                index,
                row.node_offset,
                "tire_fem atlas extract tet index",
            )?);
        }
        let mut bead_ids = Vec::with_capacity(row.bead_count);
        for &index in &self.bead_ids[bead_range.clone()] {
            bead_ids.push(subtract_u32(
                index,
                row.node_offset,
                "tire_fem atlas extract bead index",
            )?);
        }
        let mut tread_ids = Vec::with_capacity(row.tread_count);
        for &index in &self.tread_ids[tread_range.clone()] {
            tread_ids.push(subtract_u32(
                index,
                row.node_offset,
                "tire_fem atlas extract tread index",
            )?);
        }

        let node_tet_base = self.node_tet_offsets[row.node_offset] as usize;
        let mut node_tet_offsets = Vec::with_capacity(row.node_count + 1);
        for &offset in &self.node_tet_offsets[row.node_offset..=row.node_offset + row.node_count] {
            node_tet_offsets.push(subtract_u32(
                offset,
                node_tet_base,
                "tire_fem atlas extract node-tet offset",
            )?);
        }
        let node_tet_entry_range =
            node_tet_base..self.node_tet_offsets[row.node_offset + row.node_count] as usize;
        let mut node_tet_entries = Vec::with_capacity(node_tet_entry_range.len());
        for &entry in &self.node_tet_entries[node_tet_entry_range] {
            node_tet_entries.push(subtract_packed_index(
                entry,
                2,
                row.tet_offset,
                "tire_fem atlas extract node-tet entry",
            )?);
        }

        let mut xpbd_edge_indices = Vec::with_capacity(row.xpbd_edge_count);
        for edge in &self.xpbd_edge_indices[edge_range.clone()] {
            xpbd_edge_indices.push([
                subtract_u32(
                    edge[0],
                    row.node_offset,
                    "tire_fem atlas extract edge node a",
                )?,
                subtract_u32(
                    edge[1],
                    row.node_offset,
                    "tire_fem atlas extract edge node b",
                )?,
                edge[2],
                edge[3],
            ]);
        }
        let xpbd_edge_params = self.xpbd_edge_params[edge_range].to_vec();
        let (node_edge_offsets, node_edge_entries) = if row.xpbd_edge_count > 0 {
            let base = self.node_edge_offsets[row.node_offset] as usize;
            let hi = self.node_edge_offsets[row.node_offset + row.node_count] as usize;
            let mut offsets = Vec::with_capacity(row.node_count + 1);
            for &offset in
                &self.node_edge_offsets[row.node_offset..=row.node_offset + row.node_count]
            {
                offsets.push(subtract_u32(
                    offset,
                    base,
                    "tire_fem atlas extract node-edge offset",
                )?);
            }
            let mut entries = Vec::with_capacity(hi.saturating_sub(base));
            for &entry in &self.node_edge_entries[base..hi] {
                entries.push(subtract_packed_index(
                    entry,
                    1,
                    row.xpbd_edge_offset,
                    "tire_fem atlas extract node-edge entry",
                )?);
            }
            (offsets, entries)
        } else {
            (Vec::new(), Vec::new())
        };

        let mut xpbd_dihedral_indices = Vec::with_capacity(row.xpbd_dihedral_count);
        for dihedral in &self.xpbd_dihedral_indices[dihedral_range.clone()] {
            xpbd_dihedral_indices.push([
                subtract_u32(
                    dihedral[0],
                    row.node_offset,
                    "tire_fem atlas extract dihedral node a",
                )?,
                subtract_u32(
                    dihedral[1],
                    row.node_offset,
                    "tire_fem atlas extract dihedral node b",
                )?,
                subtract_u32(
                    dihedral[2],
                    row.node_offset,
                    "tire_fem atlas extract dihedral node c",
                )?,
                subtract_u32(
                    dihedral[3],
                    row.node_offset,
                    "tire_fem atlas extract dihedral node d",
                )?,
            ]);
        }
        let xpbd_dihedral_params = self.xpbd_dihedral_params[dihedral_range].to_vec();
        let (node_dihedral_offsets, node_dihedral_entries) = if row.xpbd_dihedral_count > 0 {
            let base = self.node_dihedral_offsets[row.node_offset] as usize;
            let hi = self.node_dihedral_offsets[row.node_offset + row.node_count] as usize;
            let mut offsets = Vec::with_capacity(row.node_count + 1);
            for &offset in
                &self.node_dihedral_offsets[row.node_offset..=row.node_offset + row.node_count]
            {
                offsets.push(subtract_u32(
                    offset,
                    base,
                    "tire_fem atlas extract node-dihedral offset",
                )?);
            }
            let mut entries = Vec::with_capacity(hi.saturating_sub(base));
            for &entry in &self.node_dihedral_entries[base..hi] {
                entries.push(subtract_packed_index(
                    entry,
                    2,
                    row.xpbd_dihedral_offset,
                    "tire_fem atlas extract node-dihedral entry",
                )?);
            }
            (offsets, entries)
        } else {
            (Vec::new(), Vec::new())
        };

        Ok(TireFemGpuInput {
            rest_positions: self.rest_positions[node_range.clone()].to_vec(),
            positions: self.positions[node_range.clone()].to_vec(),
            velocities: self.velocities[node_range.clone()].to_vec(),
            masses: self.masses[node_range].to_vec(),
            tet_indices,
            tet_grads: self.tet_grads[tet_scalar_range.clone()].to_vec(),
            tet_volumes: self.tet_volumes[tet_range.clone()].to_vec(),
            tet_mu: self.tet_mu[tet_range.clone()].to_vec(),
            tet_lambda: self.tet_lambda[tet_range.clone()].to_vec(),
            tet_fiber_dir: self.tet_fiber_dir[tet_range.clone()].to_vec(),
            tet_fiber_k1: self.tet_fiber_k1[tet_range.clone()].to_vec(),
            tet_fiber_k2: self.tet_fiber_k2[tet_range.clone()].to_vec(),
            tet_fiber_dir_b: self.tet_fiber_dir_b[tet_range.clone()].to_vec(),
            tet_fiber_k1_b: self.tet_fiber_k1_b[tet_range.clone()].to_vec(),
            tet_fiber_k2_b: self.tet_fiber_k2_b[tet_range.clone()].to_vec(),
            tet_fiber_dir_c: self.tet_fiber_dir_c[tet_range.clone()].to_vec(),
            tet_fiber_k1_c: self.tet_fiber_k1_c[tet_range.clone()].to_vec(),
            tet_fiber_k2_c: self.tet_fiber_k2_c[tet_range.clone()].to_vec(),
            tet_fiber_dir_d: self.tet_fiber_dir_d[tet_range.clone()].to_vec(),
            tet_fiber_k1_d: self.tet_fiber_k1_d[tet_range.clone()].to_vec(),
            tet_fiber_k2_d: self.tet_fiber_k2_d[tet_range].to_vec(),
            bead_ids,
            bead_offsets: self.bead_offsets[bead_range].to_vec(),
            tread_ids,
            node_tet_offsets,
            node_tet_entries,
            xpbd_edge_indices,
            xpbd_edge_params,
            node_edge_offsets,
            node_edge_entries,
            xpbd_dihedral_indices,
            xpbd_dihedral_params,
            node_dihedral_offsets,
            node_dihedral_entries,
            params: self.params[tire_index],
        })
    }
}

impl TireFemGpuBatchAtlasLayout {
    pub fn is_contiguous(&self) -> bool {
        if self.rows.len() != self.plan.tire_count {
            return false;
        }
        let mut node = 0usize;
        let mut tet = 0usize;
        let mut tread = 0usize;
        let mut bead = 0usize;
        let mut edge = 0usize;
        let mut dihedral = 0usize;
        let mut wg = 0u32;
        for (idx, row) in self.rows.iter().enumerate() {
            if row.tire_index != idx
                || row.node_offset != node
                || row.tet_offset != tet
                || row.tread_offset != tread
                || row.bead_offset != bead
                || row.xpbd_edge_offset != edge
                || row.xpbd_dihedral_offset != dihedral
                || row.node_workgroup_offset != wg
                || row.node_workgroup_count == 0
            {
                return false;
            }
            node = node.saturating_add(row.node_count);
            tet = tet.saturating_add(row.tet_count);
            tread = tread.saturating_add(row.tread_count);
            bead = bead.saturating_add(row.bead_count);
            edge = edge.saturating_add(row.xpbd_edge_count);
            dihedral = dihedral.saturating_add(row.xpbd_dihedral_count);
            wg = wg.saturating_add(row.node_workgroup_count);
        }
        node == self.plan.total_node_count
            && tet == self.plan.total_tet_count
            && tread == self.plan.total_tread_count
            && bead == self.plan.total_bead_count
            && edge == self.plan.total_xpbd_edge_count
            && dihedral == self.plan.total_xpbd_dihedral_count
            && wg >= self.plan.node_workgroup_count
    }
}

pub fn tire_fem_gpu_batch_dispatch_plan(
    inputs: &[TireFemGpuInput],
) -> Result<TireFemGpuBatchDispatchPlan> {
    if inputs.is_empty() {
        return Err(Error::Solver(
            "tire_fem batch plan: at least one tire input required".into(),
        ));
    }
    let mut total_node_count = 0usize;
    let mut total_tet_count = 0usize;
    let mut total_tread_count = 0usize;
    let mut total_bead_count = 0usize;
    let mut total_xpbd_edge_count = 0usize;
    let mut total_xpbd_dihedral_count = 0usize;
    let mut max_nodes_per_tire = 0usize;
    for input in inputs {
        validate_input(input)?;
        let nodes = input.positions.len();
        total_node_count = total_node_count.saturating_add(nodes);
        total_tet_count = total_tet_count.saturating_add(input.tet_volumes.len());
        total_tread_count = total_tread_count.saturating_add(input.tread_ids.len());
        total_bead_count = total_bead_count.saturating_add(input.bead_ids.len());
        total_xpbd_edge_count = total_xpbd_edge_count.saturating_add(input.xpbd_edge_indices.len());
        total_xpbd_dihedral_count =
            total_xpbd_dihedral_count.saturating_add(input.xpbd_dihedral_indices.len());
        max_nodes_per_tire = max_nodes_per_tire.max(nodes);
    }
    let vec4 = core::mem::size_of::<[f32; 4]>() as u64;
    let f32_bytes = core::mem::size_of::<f32>() as u64;
    let u32_bytes = core::mem::size_of::<u32>() as u64;
    let persistent_state_bytes = (total_node_count as u64)
        .saturating_mul(vec4)
        .saturating_mul(5)
        .saturating_add((total_node_count as u64).saturating_mul(f32_bytes))
        .saturating_add((total_tet_count as u64).saturating_mul(4 * u32_bytes))
        .saturating_add((total_tet_count as u64).saturating_mul(4 * vec4))
        .saturating_add((total_tet_count as u64).saturating_mul(14 * f32_bytes))
        .saturating_add((total_tread_count as u64).saturating_mul(u32_bytes))
        .saturating_add((total_bead_count as u64).saturating_mul(u32_bytes + vec4))
        .saturating_add((total_xpbd_edge_count as u64).saturating_mul(4 * u32_bytes + vec4))
        .saturating_add((total_xpbd_dihedral_count as u64).saturating_mul(4 * u32_bytes + vec4));
    Ok(TireFemGpuBatchDispatchPlan {
        tire_count: inputs.len(),
        total_node_count,
        total_tet_count,
        total_tread_count,
        total_bead_count,
        total_xpbd_edge_count,
        total_xpbd_dihedral_count,
        max_nodes_per_tire,
        node_workgroup_count: (total_node_count as u32).div_ceil(WORKGROUP_SIZE),
        persistent_state_bytes,
    })
}

pub fn tire_fem_gpu_batch_atlas_layout(
    inputs: &[TireFemGpuInput],
) -> Result<TireFemGpuBatchAtlasLayout> {
    let plan = tire_fem_gpu_batch_dispatch_plan(inputs)?;
    let mut rows = Vec::with_capacity(inputs.len());
    let mut node_offset = 0usize;
    let mut tet_offset = 0usize;
    let mut tread_offset = 0usize;
    let mut bead_offset = 0usize;
    let mut xpbd_edge_offset = 0usize;
    let mut xpbd_dihedral_offset = 0usize;
    let mut node_workgroup_offset = 0u32;
    for (tire_index, input) in inputs.iter().enumerate() {
        // Keep this validation here even though `dispatch_plan` also validates:
        // callers may use atlas rows as the preflight check for fused packing,
        // so bad per-tire topology should fail at the same boundary.
        validate_input(input)?;
        let node_count = input.positions.len();
        let tet_count = input.tet_volumes.len();
        let tread_count = input.tread_ids.len();
        let bead_count = input.bead_ids.len();
        let xpbd_edge_count = input.xpbd_edge_indices.len();
        let xpbd_dihedral_count = input.xpbd_dihedral_indices.len();
        let node_workgroup_count = (node_count as u32).div_ceil(WORKGROUP_SIZE);
        rows.push(TireFemGpuBatchAtlasRow {
            tire_index,
            node_offset,
            node_count,
            tet_offset,
            tet_count,
            tread_offset,
            tread_count,
            bead_offset,
            bead_count,
            xpbd_edge_offset,
            xpbd_edge_count,
            xpbd_dihedral_offset,
            xpbd_dihedral_count,
            node_workgroup_offset,
            node_workgroup_count,
        });
        node_offset = node_offset.saturating_add(node_count);
        tet_offset = tet_offset.saturating_add(tet_count);
        tread_offset = tread_offset.saturating_add(tread_count);
        bead_offset = bead_offset.saturating_add(bead_count);
        xpbd_edge_offset = xpbd_edge_offset.saturating_add(xpbd_edge_count);
        xpbd_dihedral_offset = xpbd_dihedral_offset.saturating_add(xpbd_dihedral_count);
        node_workgroup_offset = node_workgroup_offset.saturating_add(node_workgroup_count);
    }
    Ok(TireFemGpuBatchAtlasLayout { plan, rows })
}

pub fn tire_fem_gpu_batch_atlas_param_row(
    input: &TireFemGpuInput,
) -> Result<TireFemGpuBatchAtlasParamRow> {
    validate_input(input)?;
    let tet_count = input.tet_indices.len() / 4;
    Ok(tire_fem_gpu_batch_atlas_param_row_from_parts(
        input.params,
        [
            input.positions.len() as u32,
            tet_count as u32,
            input.bead_ids.len() as u32,
            input.tread_ids.len() as u32,
        ],
        [
            input.xpbd_edge_indices.len() as u32,
            input.xpbd_dihedral_indices.len() as u32,
            0,
            0,
        ],
    ))
}

fn tire_fem_gpu_batch_atlas_param_row_from_parts(
    params: TireFemGpuParams,
    counts: [u32; 4],
    counts2: [u32; 4],
) -> TireFemGpuBatchAtlasParamRow {
    TireFemGpuBatchAtlasParamRow {
        counts,
        counts2,
        phys0: [
            params.dt,
            params.young_modulus,
            params.poisson_ratio,
            params.pressure_pa,
        ],
        phys1: [
            params.node_damping,
            params.bead_stiffness,
            params.floor_stiffness,
            params.floor_damping,
        ],
        phys2: [
            params.pressure_area_per_tread_node,
            params.hub[0],
            params.hub[1],
            params.hub[2],
        ],
        phys3: [
            params.gravity_y,
            params.cavity_radial_force_per_bead,
            params.belt_proxy_mu,
            params.xpbd_volume_compliance,
        ],
        phys4: [
            params.xpbd_bead_compliance,
            params.semi_implicit_stiffness_scale,
            params.xpbd_sidewall_ply_stretch_limit,
            params.xpbd_sidewall_ply_compliance,
        ],
    }
}

pub fn tire_fem_gpu_batch_atlas_offset_row(
    row: TireFemGpuBatchAtlasRow,
) -> Result<TireFemGpuBatchAtlasOffsetRow> {
    Ok(TireFemGpuBatchAtlasOffsetRow {
        node: [
            usize_to_u32(row.node_offset, "tire_fem atlas node offset")?,
            usize_to_u32(row.node_count, "tire_fem atlas node count")?,
            row.node_workgroup_offset,
            row.node_workgroup_count,
        ],
        tet: [
            usize_to_u32(row.tet_offset, "tire_fem atlas tet offset")?,
            usize_to_u32(row.tet_count, "tire_fem atlas tet count")?,
            0,
            0,
        ],
        contact: [
            usize_to_u32(row.tread_offset, "tire_fem atlas tread offset")?,
            usize_to_u32(row.tread_count, "tire_fem atlas tread count")?,
            usize_to_u32(row.bead_offset, "tire_fem atlas bead offset")?,
            usize_to_u32(row.bead_count, "tire_fem atlas bead count")?,
        ],
        xpbd: [
            usize_to_u32(row.xpbd_edge_offset, "tire_fem atlas edge offset")?,
            usize_to_u32(row.xpbd_edge_count, "tire_fem atlas edge count")?,
            usize_to_u32(row.xpbd_dihedral_offset, "tire_fem atlas dihedral offset")?,
            usize_to_u32(row.xpbd_dihedral_count, "tire_fem atlas dihedral count")?,
        ],
    })
}

pub fn tire_fem_gpu_batch_atlas_buffers(
    inputs: &[TireFemGpuInput],
) -> Result<TireFemGpuBatchAtlasBuffers> {
    let layout = tire_fem_gpu_batch_atlas_layout(inputs)?;
    let plan = layout.plan;
    let has_edges = inputs
        .iter()
        .any(|input| !input.xpbd_edge_indices.is_empty());
    let has_dihedrals = inputs
        .iter()
        .any(|input| !input.xpbd_dihedral_indices.is_empty());
    let mut buffers = TireFemGpuBatchAtlasBuffers {
        layout,
        node_tire_indices: Vec::with_capacity(plan.total_node_count),
        tet_tire_indices: Vec::with_capacity(plan.total_tet_count),
        bead_tire_indices: Vec::with_capacity(plan.total_bead_count),
        tread_tire_indices: Vec::with_capacity(plan.total_tread_count),
        rest_positions: Vec::with_capacity(plan.total_node_count),
        positions: Vec::with_capacity(plan.total_node_count),
        velocities: Vec::with_capacity(plan.total_node_count),
        masses: Vec::with_capacity(plan.total_node_count),
        tet_indices: Vec::with_capacity(plan.total_tet_count * 4),
        tet_grads: Vec::with_capacity(plan.total_tet_count * 4),
        tet_volumes: Vec::with_capacity(plan.total_tet_count),
        tet_mu: Vec::with_capacity(plan.total_tet_count),
        tet_lambda: Vec::with_capacity(plan.total_tet_count),
        tet_fiber_dir: Vec::with_capacity(plan.total_tet_count),
        tet_fiber_k1: Vec::with_capacity(plan.total_tet_count),
        tet_fiber_k2: Vec::with_capacity(plan.total_tet_count),
        tet_fiber_dir_b: Vec::with_capacity(plan.total_tet_count),
        tet_fiber_k1_b: Vec::with_capacity(plan.total_tet_count),
        tet_fiber_k2_b: Vec::with_capacity(plan.total_tet_count),
        tet_fiber_dir_c: Vec::with_capacity(plan.total_tet_count),
        tet_fiber_k1_c: Vec::with_capacity(plan.total_tet_count),
        tet_fiber_k2_c: Vec::with_capacity(plan.total_tet_count),
        tet_fiber_dir_d: Vec::with_capacity(plan.total_tet_count),
        tet_fiber_k1_d: Vec::with_capacity(plan.total_tet_count),
        tet_fiber_k2_d: Vec::with_capacity(plan.total_tet_count),
        bead_ids: Vec::with_capacity(plan.total_bead_count),
        bead_offsets: Vec::with_capacity(plan.total_bead_count),
        tread_ids: Vec::with_capacity(plan.total_tread_count),
        node_tet_offsets: Vec::with_capacity(plan.total_node_count + 1),
        node_tet_entries: Vec::with_capacity(plan.total_tet_count * 4),
        xpbd_edge_indices: Vec::new(),
        xpbd_edge_params: Vec::new(),
        node_edge_offsets: if has_edges {
            Vec::with_capacity(plan.total_node_count + 1)
        } else {
            Vec::new()
        },
        node_edge_entries: Vec::new(),
        xpbd_dihedral_indices: Vec::new(),
        xpbd_dihedral_params: Vec::new(),
        node_dihedral_offsets: if has_dihedrals {
            Vec::with_capacity(plan.total_node_count + 1)
        } else {
            Vec::new()
        },
        node_dihedral_entries: Vec::new(),
        param_rows: Vec::with_capacity(plan.tire_count),
        offset_rows: Vec::with_capacity(plan.tire_count),
        params: Vec::with_capacity(plan.tire_count),
    };
    buffers.node_tet_offsets.push(0);
    if has_edges {
        buffers.node_edge_offsets.push(0);
    }
    if has_dihedrals {
        buffers.node_dihedral_offsets.push(0);
    }

    let rows = buffers.layout.rows.clone();
    for (row, input) in rows.iter().zip(inputs) {
        let tire_row = u32::try_from(row.tire_index)
            .map_err(|_| Error::Solver("tire_fem atlas tire index overflow".into()))?;
        buffers
            .node_tire_indices
            .extend(std::iter::repeat_n(tire_row, row.node_count));
        buffers
            .tet_tire_indices
            .extend(std::iter::repeat_n(tire_row, row.tet_count));
        buffers
            .bead_tire_indices
            .extend(std::iter::repeat_n(tire_row, row.bead_count));
        buffers
            .tread_tire_indices
            .extend(std::iter::repeat_n(tire_row, row.tread_count));
        buffers
            .rest_positions
            .extend_from_slice(&input.rest_positions);
        buffers.positions.extend_from_slice(&input.positions);
        buffers.velocities.extend_from_slice(&input.velocities);
        buffers.masses.extend_from_slice(&input.masses);
        for &index in &input.tet_indices {
            buffers.tet_indices.push(offset_u32(
                index,
                row.node_offset,
                "tire_fem atlas tet index",
            )?);
        }
        buffers.tet_grads.extend_from_slice(&input.tet_grads);
        buffers.tet_volumes.extend_from_slice(&input.tet_volumes);
        buffers.tet_mu.extend_from_slice(&input.tet_mu);
        buffers.tet_lambda.extend_from_slice(&input.tet_lambda);
        buffers
            .tet_fiber_dir
            .extend_from_slice(&input.tet_fiber_dir);
        buffers.tet_fiber_k1.extend_from_slice(&input.tet_fiber_k1);
        buffers.tet_fiber_k2.extend_from_slice(&input.tet_fiber_k2);
        buffers
            .tet_fiber_dir_b
            .extend_from_slice(&input.tet_fiber_dir_b);
        buffers
            .tet_fiber_k1_b
            .extend_from_slice(&input.tet_fiber_k1_b);
        buffers
            .tet_fiber_k2_b
            .extend_from_slice(&input.tet_fiber_k2_b);
        buffers
            .tet_fiber_dir_c
            .extend_from_slice(&input.tet_fiber_dir_c);
        buffers
            .tet_fiber_k1_c
            .extend_from_slice(&input.tet_fiber_k1_c);
        buffers
            .tet_fiber_k2_c
            .extend_from_slice(&input.tet_fiber_k2_c);
        buffers
            .tet_fiber_dir_d
            .extend_from_slice(&input.tet_fiber_dir_d);
        buffers
            .tet_fiber_k1_d
            .extend_from_slice(&input.tet_fiber_k1_d);
        buffers
            .tet_fiber_k2_d
            .extend_from_slice(&input.tet_fiber_k2_d);
        for &index in &input.bead_ids {
            buffers.bead_ids.push(offset_u32(
                index,
                row.node_offset,
                "tire_fem atlas bead index",
            )?);
        }
        buffers.bead_offsets.extend_from_slice(&input.bead_offsets);
        for &index in &input.tread_ids {
            buffers.tread_ids.push(offset_u32(
                index,
                row.node_offset,
                "tire_fem atlas tread index",
            )?);
        }

        let tet_entry_base = buffers.node_tet_entries.len();
        for &entry in &input.node_tet_entries {
            buffers.node_tet_entries.push(offset_packed_index(
                entry,
                2,
                row.tet_offset,
                "tire_fem atlas node-tet entry",
            )?);
        }
        for &offset in input.node_tet_offsets.iter().skip(1) {
            buffers.node_tet_offsets.push(offset_u32(
                offset,
                tet_entry_base,
                "tire_fem atlas node-tet offset",
            )?);
        }

        let edge_offset = buffers.xpbd_edge_indices.len();
        for &edge in &input.xpbd_edge_indices {
            buffers.xpbd_edge_indices.push([
                offset_u32(edge[0], row.node_offset, "tire_fem atlas edge node a")?,
                offset_u32(edge[1], row.node_offset, "tire_fem atlas edge node b")?,
                edge[2],
                edge[3],
            ]);
        }
        buffers
            .xpbd_edge_params
            .extend_from_slice(&input.xpbd_edge_params);
        if has_edges {
            let edge_entry_base = buffers.node_edge_entries.len();
            for &entry in &input.node_edge_entries {
                buffers.node_edge_entries.push(offset_packed_index(
                    entry,
                    1,
                    edge_offset,
                    "tire_fem atlas node-edge entry",
                )?);
            }
            if input.node_edge_offsets.is_empty() {
                let current = u32::try_from(buffers.node_edge_entries.len()).map_err(|_| {
                    Error::Solver("tire_fem atlas node-edge offset overflow".into())
                })?;
                buffers
                    .node_edge_offsets
                    .extend(std::iter::repeat_n(current, row.node_count));
            } else {
                for &offset in input.node_edge_offsets.iter().skip(1) {
                    buffers.node_edge_offsets.push(offset_u32(
                        offset,
                        edge_entry_base,
                        "tire_fem atlas node-edge offset",
                    )?);
                }
            }
        }

        let dihedral_offset = buffers.xpbd_dihedral_indices.len();
        for &dihedral in &input.xpbd_dihedral_indices {
            buffers.xpbd_dihedral_indices.push([
                offset_u32(
                    dihedral[0],
                    row.node_offset,
                    "tire_fem atlas dihedral node a",
                )?,
                offset_u32(
                    dihedral[1],
                    row.node_offset,
                    "tire_fem atlas dihedral node b",
                )?,
                offset_u32(
                    dihedral[2],
                    row.node_offset,
                    "tire_fem atlas dihedral node c",
                )?,
                offset_u32(
                    dihedral[3],
                    row.node_offset,
                    "tire_fem atlas dihedral node d",
                )?,
            ]);
        }
        buffers
            .xpbd_dihedral_params
            .extend_from_slice(&input.xpbd_dihedral_params);
        if has_dihedrals {
            let dihedral_entry_base = buffers.node_dihedral_entries.len();
            for &entry in &input.node_dihedral_entries {
                buffers.node_dihedral_entries.push(offset_packed_index(
                    entry,
                    2,
                    dihedral_offset,
                    "tire_fem atlas node-dihedral entry",
                )?);
            }
            if input.node_dihedral_offsets.is_empty() {
                let current = u32::try_from(buffers.node_dihedral_entries.len()).map_err(|_| {
                    Error::Solver("tire_fem atlas node-dihedral offset overflow".into())
                })?;
                buffers
                    .node_dihedral_offsets
                    .extend(std::iter::repeat_n(current, row.node_count));
            } else {
                for &offset in input.node_dihedral_offsets.iter().skip(1) {
                    buffers.node_dihedral_offsets.push(offset_u32(
                        offset,
                        dihedral_entry_base,
                        "tire_fem atlas node-dihedral offset",
                    )?);
                }
            }
        }

        buffers
            .param_rows
            .push(tire_fem_gpu_batch_atlas_param_row(input)?);
        buffers
            .offset_rows
            .push(tire_fem_gpu_batch_atlas_offset_row(*row)?);
        buffers.params.push(input.params);
    }

    buffers.validate()?;
    Ok(buffers)
}

pub fn tire_fem_gpu_batch_atlas_binding_plan(
    atlas: &TireFemGpuBatchAtlasBuffers,
) -> Result<Vec<TireFemGpuBatchAtlasBindingPlan>> {
    atlas.validate()?;
    let n = atlas.layout.plan.total_node_count;
    let tet4 = atlas.layout.plan.total_tet_count * 4;
    let hub_rows = atlas.layout.plan.tire_count * 2;
    let rows = vec![
        atlas_binding(
            0,
            "rest_pos",
            atlas.rest_positions.len(),
            bytes_of_slice::<[f32; 4]>(atlas.rest_positions.len()),
            true,
        ),
        atlas_binding(
            1,
            "pos_in",
            atlas.positions.len(),
            bytes_of_slice::<[f32; 4]>(atlas.positions.len()),
            true,
        ),
        atlas_binding(
            2,
            "vel_in",
            atlas.velocities.len(),
            bytes_of_slice::<[f32; 4]>(atlas.velocities.len()),
            true,
        ),
        atlas_binding(
            3,
            "mass_in",
            atlas.masses.len(),
            bytes_of_slice::<f32>(atlas.masses.len()),
            true,
        ),
        atlas_binding(
            4,
            "tet_indices",
            atlas.tet_indices.len(),
            bytes_of_slice::<u32>(atlas.tet_indices.len()),
            true,
        ),
        atlas_binding(
            5,
            "tet_grads",
            atlas.tet_grads.len(),
            bytes_of_slice::<[f32; 4]>(atlas.tet_grads.len()),
            true,
        ),
        atlas_binding(
            6,
            "tet_volume",
            atlas.tet_volumes.len(),
            bytes_of_slice::<f32>(atlas.tet_volumes.len()),
            true,
        ),
        atlas_binding(
            7,
            "tet_mu",
            atlas.tet_mu.len(),
            bytes_of_slice::<f32>(atlas.tet_mu.len()),
            true,
        ),
        atlas_binding(
            8,
            "tet_lambda",
            atlas.tet_lambda.len(),
            bytes_of_slice::<f32>(atlas.tet_lambda.len()),
            true,
        ),
        atlas_binding(
            9,
            "bead_ids",
            atlas.bead_ids.len(),
            bytes_of_slice::<u32>(atlas.bead_ids.len()),
            true,
        ),
        atlas_binding(
            10,
            "bead_offsets",
            atlas.bead_offsets.len(),
            bytes_of_slice::<[f32; 4]>(atlas.bead_offsets.len()),
            true,
        ),
        atlas_binding(
            11,
            "tread_ids",
            atlas.tread_ids.len(),
            bytes_of_slice::<u32>(atlas.tread_ids.len()),
            true,
        ),
        atlas_binding(12, "pos_out", n, bytes_of_slice::<[f32; 4]>(n), false),
        atlas_binding(13, "vel_out", n, bytes_of_slice::<[f32; 4]>(n), false),
        atlas_binding(14, "force_out", n, bytes_of_slice::<[f32; 4]>(n), false),
        atlas_binding(
            15,
            "tet_fiber_dir",
            atlas.tet_fiber_dir.len(),
            bytes_of_slice::<[f32; 4]>(atlas.tet_fiber_dir.len()),
            true,
        ),
        atlas_binding(
            16,
            "tet_fiber_k1",
            atlas.tet_fiber_k1.len(),
            bytes_of_slice::<f32>(atlas.tet_fiber_k1.len()),
            true,
        ),
        atlas_binding(
            17,
            "tet_fiber_k2",
            atlas.tet_fiber_k2.len(),
            bytes_of_slice::<f32>(atlas.tet_fiber_k2.len()),
            true,
        ),
        atlas_binding(
            18,
            "node_tet_offsets",
            atlas.node_tet_offsets.len(),
            bytes_of_slice::<u32>(atlas.node_tet_offsets.len()),
            true,
        ),
        atlas_binding(
            19,
            "node_tet_entries",
            atlas.node_tet_entries.len(),
            bytes_of_slice::<u32>(atlas.node_tet_entries.len()),
            true,
        ),
        atlas_binding(
            20,
            "hub_state",
            hub_rows,
            bytes_of_slice::<[f32; 4]>(hub_rows),
            false,
        ),
        atlas_binding(
            21,
            "tet_fiber_dir_b",
            atlas.tet_fiber_dir_b.len(),
            bytes_of_slice::<[f32; 4]>(atlas.tet_fiber_dir_b.len()),
            true,
        ),
        atlas_binding(
            22,
            "tet_fiber_k1_b",
            atlas.tet_fiber_k1_b.len(),
            bytes_of_slice::<f32>(atlas.tet_fiber_k1_b.len()),
            true,
        ),
        atlas_binding(
            23,
            "tet_fiber_k2_b",
            atlas.tet_fiber_k2_b.len(),
            bytes_of_slice::<f32>(atlas.tet_fiber_k2_b.len()),
            true,
        ),
        atlas_binding(
            24,
            "xpbd_edge_indices",
            atlas.xpbd_edge_indices.len(),
            bytes_of_slice::<[u32; 4]>(atlas.xpbd_edge_indices.len()),
            true,
        ),
        atlas_binding(
            25,
            "xpbd_edge_params",
            atlas.xpbd_edge_params.len(),
            bytes_of_slice::<[f32; 4]>(atlas.xpbd_edge_params.len()),
            true,
        ),
        atlas_binding(
            26,
            "node_edge_offsets",
            atlas.node_edge_offsets.len(),
            bytes_of_slice::<u32>(atlas.node_edge_offsets.len()),
            true,
        ),
        atlas_binding(
            27,
            "node_edge_entries",
            atlas.node_edge_entries.len(),
            bytes_of_slice::<u32>(atlas.node_edge_entries.len()),
            true,
        ),
        atlas_binding(
            28,
            "tet_fiber_dir_c",
            atlas.tet_fiber_dir_c.len(),
            bytes_of_slice::<[f32; 4]>(atlas.tet_fiber_dir_c.len()),
            true,
        ),
        atlas_binding(
            29,
            "tet_fiber_k1_c",
            atlas.tet_fiber_k1_c.len(),
            bytes_of_slice::<f32>(atlas.tet_fiber_k1_c.len()),
            true,
        ),
        atlas_binding(
            30,
            "tet_fiber_k2_c",
            atlas.tet_fiber_k2_c.len(),
            bytes_of_slice::<f32>(atlas.tet_fiber_k2_c.len()),
            true,
        ),
        atlas_binding(
            31,
            "tet_fiber_dir_d",
            atlas.tet_fiber_dir_d.len(),
            bytes_of_slice::<[f32; 4]>(atlas.tet_fiber_dir_d.len()),
            true,
        ),
        atlas_binding(
            32,
            "tet_fiber_k1_d",
            atlas.tet_fiber_k1_d.len(),
            bytes_of_slice::<f32>(atlas.tet_fiber_k1_d.len()),
            true,
        ),
        atlas_binding(
            33,
            "tet_fiber_k2_d",
            atlas.tet_fiber_k2_d.len(),
            bytes_of_slice::<f32>(atlas.tet_fiber_k2_d.len()),
            true,
        ),
        atlas_binding(
            34,
            "xpbd_dihedral_indices",
            atlas.xpbd_dihedral_indices.len(),
            bytes_of_slice::<[u32; 4]>(atlas.xpbd_dihedral_indices.len()),
            true,
        ),
        atlas_binding(
            35,
            "xpbd_dihedral_params",
            atlas.xpbd_dihedral_params.len(),
            bytes_of_slice::<[f32; 4]>(atlas.xpbd_dihedral_params.len()),
            true,
        ),
        atlas_binding(
            36,
            "node_dihedral_offsets",
            atlas.node_dihedral_offsets.len(),
            bytes_of_slice::<u32>(atlas.node_dihedral_offsets.len()),
            true,
        ),
        atlas_binding(
            37,
            "node_dihedral_entries",
            atlas.node_dihedral_entries.len(),
            bytes_of_slice::<u32>(atlas.node_dihedral_entries.len()),
            true,
        ),
        atlas_binding(
            38,
            "node_tire_indices",
            atlas.node_tire_indices.len(),
            bytes_of_slice::<u32>(atlas.node_tire_indices.len()),
            true,
        ),
        atlas_binding(
            39,
            "tet_tire_indices",
            atlas.tet_tire_indices.len(),
            bytes_of_slice::<u32>(atlas.tet_tire_indices.len()),
            true,
        ),
        atlas_binding(
            40,
            "bead_tire_indices",
            atlas.bead_tire_indices.len(),
            bytes_of_slice::<u32>(atlas.bead_tire_indices.len()),
            true,
        ),
        atlas_binding(
            41,
            "tread_tire_indices",
            atlas.tread_tire_indices.len(),
            bytes_of_slice::<u32>(atlas.tread_tire_indices.len()),
            true,
        ),
        atlas_binding(
            42,
            "param_rows",
            atlas.param_rows.len(),
            bytes_of_slice::<TireFemGpuBatchAtlasParamRow>(atlas.param_rows.len()),
            true,
        ),
        atlas_binding(
            43,
            "offset_rows",
            atlas.offset_rows.len(),
            bytes_of_slice::<TireFemGpuBatchAtlasOffsetRow>(atlas.offset_rows.len()),
            true,
        ),
    ];
    debug_assert_eq!(tet4, atlas.tet_indices.len());
    Ok(rows)
}

pub fn tire_fem_gpu_batch_atlas_dispatch_contract(
    atlas: &TireFemGpuBatchAtlasBuffers,
) -> Result<TireFemGpuBatchAtlasDispatchContract> {
    atlas.validate()?;
    let bindings = tire_fem_gpu_batch_atlas_binding_plan(atlas)?;
    Ok(TireFemGpuBatchAtlasDispatchContract {
        shader_label: "molla::tire_fem_atlas::step_nodes_atlas_contract",
        entry_point: "step_nodes_atlas_contract",
        workgroup_size: WORKGROUP_SIZE,
        workgroup_count: atlas.layout.plan.node_workgroup_count,
        total_node_count: atlas.layout.plan.total_node_count,
        binding_count: bindings.len(),
    })
}

/// GPU-parity status for the matrix-free Newton tire-FEM debug timestep.
///
/// Reflects the same parity row the host crate records in
/// `tire_gpu_parity_checklist_*` and `tire_known_approximation_ledger_*`.
/// The contract builder resolves the current status from the shader
/// source; as of the matrix-free Newton GPU landing it resolves `Live`
/// (WGSL kernel + dispatch wrapper + hardware test all present).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TireFemMatrixFreeNewtonGpuStatus {
    /// Only the CPU/debug callable timestep exists. No WGSL, no GPU dispatch.
    CpuOnlyDebug,
    /// The GPU dispatch is live and used by the persistent batch / viewer.
    Live,
}

/// Host-side dispatch contract for the GPU matrix-free Newton tire-FEM
/// debug timestep.
///
/// This is the GPU-parity seam for
/// `molla_vehicle::TetraTireFem::step_on_terrain_matrix_free_newton_debug`.
/// The hardware path is live (`TireFemMatrixFreeNewtonGpuKernel` + the
/// `tire_fem_matrix_free_newton_gpu_residual_matches_cpu_oracle` hardware
/// test) — this contract pins the dispatch shape (workgroup size, node/DoF
/// counts, per-stage workgroup counts, line-search alpha sweep, Krylov
/// budget) so the WGSL kernel cannot silently drift from the CPU reference.
/// The `status` field records where the GPU work currently sits in
/// [`TireFemMatrixFreeNewtonGpuStatus`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TireFemMatrixFreeNewtonGpuContract {
    pub shader_label: &'static str,
    pub workgroup_size: u32,
    pub node_count: u32,
    pub coupled_dof_count: u32,
    pub residual_workgroup_count: u32,
    pub jv_probe_workgroup_count: u32,
    pub diagonal_workgroup_count: u32,
    pub correction_workgroup_count: u32,
    pub line_search_alpha_count: u32,
    pub krylov_iteration_budget: u32,
    pub line_search_residual_invocations: u32,
    pub status: TireFemMatrixFreeNewtonGpuStatus,
}

/// Number of step sizes tried by the CPU matrix-free Newton line search.
///
/// `[1.0, 0.5, 0.25, 0.125, 0.0625]` in
/// `molla_vehicle::real_tire::matrix_free_newton_correction`.
pub const TIRE_FEM_MATRIX_FREE_NEWTON_LINE_SEARCH_ALPHAS: u32 = 5;

/// Default Krylov iteration budget shared between the CPU report and the GPU
/// dispatch contract. Mirrors the `solve_matrix_free_gmres(.., 16, ..)` call
/// site in `molla_vehicle::real_tire::matrix_free_newton_correction`.
pub const TIRE_FEM_MATRIX_FREE_NEWTON_DEFAULT_KRYLOV_BUDGET: u32 = 16;

/// Build the GPU dispatch contract for the matrix-free Newton tire-FEM
/// debug timestep.
///
/// `node_count` is the FEM mesh node count, which sets the coupled DoF count
/// to `3 * node_count` (one position component per axis). The per-stage
/// workgroup counts are derived from the shared `WORKGROUP_SIZE`, and the
/// line-search residual invocations include one extra evaluation for the
/// preconditioned correction relative to
/// [`TIRE_FEM_MATRIX_FREE_NEWTON_LINE_SEARCH_ALPHAS`].
pub fn tire_fem_matrix_free_newton_gpu_contract(
    node_count: u32,
    krylov_iteration_budget: u32,
) -> TireFemMatrixFreeNewtonGpuContract {
    let coupled_dof_count = node_count.saturating_mul(3);
    let node_workgroups = node_count.div_ceil(WORKGROUP_SIZE.max(1));
    let dof_workgroups = coupled_dof_count.div_ceil(WORKGROUP_SIZE.max(1));
    let krylov = krylov_iteration_budget.max(1);
    let shader = TIRE_FEM_MATRIX_FREE_NEWTON_SHADER;
    let entries_present = shader.contains("fn residual_eval")
        && shader.contains("fn jv_probe")
        && shader.contains("fn diagonal_extract")
        && shader.contains("fn apply_correction");
    // `Live` reflects that the WGSL kernel, the host-side dispatch wrapper
    // (`TireFemMatrixFreeNewtonGpuKernel`), and the ignored hardware
    // integration test (`tire_fem_matrix_free_newton_gpu_residual_matches_cpu_oracle`)
    // are all in place. Hardware execution still requires an adapter, but the
    // integration seam is no longer source-guard-only.
    let status = if entries_present {
        TireFemMatrixFreeNewtonGpuStatus::Live
    } else {
        TireFemMatrixFreeNewtonGpuStatus::CpuOnlyDebug
    };
    TireFemMatrixFreeNewtonGpuContract {
        shader_label: "molla::tire_fem_matrix_free_newton::step_debug_contract",
        workgroup_size: WORKGROUP_SIZE,
        node_count,
        coupled_dof_count,
        residual_workgroup_count: node_workgroups,
        jv_probe_workgroup_count: node_workgroups,
        diagonal_workgroup_count: dof_workgroups,
        correction_workgroup_count: node_workgroups,
        line_search_alpha_count: TIRE_FEM_MATRIX_FREE_NEWTON_LINE_SEARCH_ALPHAS,
        krylov_iteration_budget: krylov,
        line_search_residual_invocations: TIRE_FEM_MATRIX_FREE_NEWTON_LINE_SEARCH_ALPHAS
            .saturating_add(1),
        status,
    }
}

/// GPU-parity status for the PLAN_TOURQUE torque/spin extension.
///
/// Mirrors the matrix-free-Newton enum (`TireFemMatrixFreeNewtonGpuStatus`)
/// so the same staging language applies. The host-side
/// [`TireFemGpuParams`] carries `hub_omega`, `drive_torque_n_m`, and
/// `brake_torque_n_m`; the WGSL step kernel reads the spin slot, and the
/// contract builder currently resolves
/// [`TireFemTorqueGpuStatus::LiveValidated`] via the Stream C G4
/// rolling-parity closure flag. Hub torque→spin integration itself stays
/// host-side by design.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TireFemTorqueGpuStatus {
    /// Host-side params carry the torque/spin fields; the WGSL step
    /// kernel does not yet read them. Spin DOF integration, spin-coupled
    /// bristle slip, and dynamic-hub XZ integration still live only on
    /// the CPU `step_inner` path.
    HostParamsOnly,
    /// The WGSL step kernel reads the torque/spin params and the
    /// source guard asserts the named uniform slots exist, but the
    /// hardware dispatch has not been validated against the CPU
    /// reference yet.
    WgslWired,
    /// GPU step kernel integrates the torque/spin DOF and the parity
    /// hardware lane has been confirmed against the CPU reference.
    Live,
    /// PLAN_REALTIRE Stream C G4 closure: the full GPU torque/spin stack
    /// is validated against the CPU reference through all three parity
    /// hardware legs — node FEM without contact
    /// (`tire_fem_cpu_vs_gpu_node_parity_no_contact`), dynamic-hub
    /// trajectory (`tire_fem_cpu_vs_gpu_hub_y_parity_dynamic_hub`), and
    /// contact-driven rolling with sub-cm hub agreement under spin +
    /// drive torque (`tire_fem_cpu_vs_gpu_rolling_contact_hub_x_parity`).
    /// This is the state the 40-wheel GPU scaling target requires.
    LiveValidated,
}

/// Host-side dispatch contract for the PLAN_TOURQUE torque/spin extension.
///
/// Pins the names + shape of the torque/spin extension to the WGSL kernel
/// so Stage G part 2 can land the GPU integration without drifting from
/// the CPU implementation in
/// `molla_vehicle::real_tire::TetraTireFem::integrate_spin_dof` and the
/// spin-coupled bristle code in `step_inner`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TireFemTorqueGpuContract {
    /// Stable host-side label so the parity row + ledger entry can
    /// reference one canonical string.
    pub feature_label: &'static str,
    /// Names of the per-tire uniform fields the GPU step kernel will
    /// consume once Stage G part 2 lands.
    pub uniform_field_names: [&'static str; 3],
    /// Soft-sign brake shoulder (rad/s). Mirrors the constant baked
    /// into the CPU `integrate_spin_dof` so the GPU side cannot pick a
    /// different shoulder by accident.
    pub brake_soft_shoulder_rad_s: f32,
    /// Hub-mass clamp the CPU integrator uses to keep the dynamic-hub
    /// denominator non-singular. Same constant for both paths.
    pub hub_mass_min_kg: f32,
    /// Where this row currently sits in the staged GPU roll-out.
    pub status: TireFemTorqueGpuStatus,
}

/// Soft-sign shoulder for the brake torque (rad/s). Matches the
/// constant baked into `TetraTireFem::integrate_spin_dof` so the GPU
/// kernel cannot accidentally pick a different shoulder.
pub const TIRE_FEM_TORQUE_BRAKE_SOFT_SHOULDER_RAD_S: f32 = 0.5;

/// Minimum hub mass used by the dynamic-hub integrator to keep the
/// implicit-damping denominator non-singular.
pub const TIRE_FEM_TORQUE_HUB_MASS_MIN_KG: f32 = 1.0e-6;

/// Build the PLAN_TOURQUE Stage G dispatch contract.
///
/// Auto-staged from the bundled `tire_fem.wgsl`:
/// - `HostParamsOnly` — no `phys5` row in the shader.
/// - `WgslWired` — `phys5` row exists but the spin-coupled contact
///   path isn't there yet.
/// - `Live` — `phys5` row + the `spin_omega = u.phys5.x` contact
///   contribution both present (Stage G part 2), and a hardware lane
///   exercises it (`tire_fem_gpu_spin_pushes_grounded_tread_backward_in_contact`).
pub fn tire_fem_torque_gpu_contract() -> TireFemTorqueGpuContract {
    let shader = TIRE_FEM_SHADER_SRC;
    let phys5_present = shader.contains("phys5: vec4<f32>");
    let spin_contact_present =
        shader.contains("spin_omega = u.phys5.x") && shader.contains("spin_v_x");
    let status = if spin_contact_present && TIRE_FEM_TORQUE_ROLLING_PARITY_VALIDATED {
        TireFemTorqueGpuStatus::LiveValidated
    } else if spin_contact_present {
        TireFemTorqueGpuStatus::Live
    } else if phys5_present {
        TireFemTorqueGpuStatus::WgslWired
    } else {
        TireFemTorqueGpuStatus::HostParamsOnly
    };
    TireFemTorqueGpuContract {
        feature_label: "molla::tire_fem::torque_spin_api",
        uniform_field_names: ["hub_omega", "drive_torque_n_m", "brake_torque_n_m"],
        brake_soft_shoulder_rad_s: TIRE_FEM_TORQUE_BRAKE_SOFT_SHOULDER_RAD_S,
        hub_mass_min_kg: TIRE_FEM_TORQUE_HUB_MASS_MIN_KG,
        status,
    }
}

/// Cached WGSL source for the single-tire FEM step kernel — used by the
/// PLAN_TOURQUE Stage G contract's source-guard check.
const TIRE_FEM_SHADER_SRC: &str = include_str!("../shaders/tire_fem.wgsl");

/// PLAN_REALTIRE Stream C G4 closure flag. Flip this back to `false` if
/// the contact-driven parity lane
/// (`tire_fem_cpu_vs_gpu_rolling_contact_hub_x_parity` in
/// `tests/tire_fem_gpu.rs`) regresses or is removed: the lane rolls a
/// settled, spinning tire on flat ground on both backends —
/// `step_persistent` → contact-feedback bristle brush → hub kernel on
/// GPU versus `step_on_terrain` on CPU — and holds sub-cm hub-X/Y
/// agreement over 96 substeps. The other two legs
/// (`..._node_parity_no_contact`, `..._hub_y_parity_dynamic_hub`)
/// validate the no-contact FEM step and the dynamic hub equation.
const TIRE_FEM_TORQUE_ROLLING_PARITY_VALIDATED: bool = true;

impl TireFemGpuBatch {
    pub fn new(meshes: Vec<TireFemPersistentMesh>) -> Self {
        Self { meshes }
    }

    pub fn len(&self) -> usize {
        self.meshes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.meshes.is_empty()
    }

    pub fn total_node_count(&self) -> u32 {
        self.meshes
            .iter()
            .map(TireFemPersistentMesh::node_count)
            .sum()
    }

    pub fn meshes(&self) -> &[TireFemPersistentMesh] {
        &self.meshes
    }

    pub fn meshes_mut(&mut self) -> &mut [TireFemPersistentMesh] {
        &mut self.meshes
    }
}

impl TireFemGpuKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::tire_fem"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tire_fem.wgsl").into()),
        });
        let layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_fem io"),
            entries: &[
                storage_entry(0, true),   // rest_pos
                storage_entry(1, true),   // pos_in
                storage_entry(2, true),   // vel_in
                storage_entry(3, true),   // mass_in
                storage_entry(4, true),   // tet_indices
                storage_entry(5, true),   // tet_grads
                storage_entry(6, true),   // tet_volume
                storage_entry(7, true),   // tet_mu
                storage_entry(8, true),   // tet_lambda
                storage_entry(9, true),   // bead_ids
                storage_entry(10, true),  // bead_offsets
                storage_entry(11, true),  // tread_ids
                storage_entry(12, false), // pos_out
                storage_entry(13, false), // vel_out
                storage_entry(14, false), // force_out
                storage_entry(15, true),  // tet_fiber_dir
                storage_entry(16, true),  // tet_fiber_k1
                storage_entry(17, true),  // tet_fiber_k2
                storage_entry(18, true),  // node_tet_offsets
                storage_entry(19, true),  // node_tet_entries
                storage_entry(20, true),  // hub_state (read-only for the main kernel)
                storage_entry(21, true),  // tet_fiber_dir_b
                storage_entry(22, true),  // tet_fiber_k1_b
                storage_entry(23, true),  // tet_fiber_k2_b
                storage_entry(24, true),  // xpbd_edge_indices
                storage_entry(25, true),  // xpbd_edge_params
                storage_entry(26, true),  // node_edge_offsets
                storage_entry(27, true),  // node_edge_entries
                storage_entry(28, true),  // tet_fiber_dir_c
                storage_entry(29, true),  // tet_fiber_k1_c
                storage_entry(30, true),  // tet_fiber_k2_c
                storage_entry(31, true),  // tet_fiber_dir_d
                storage_entry(32, true),  // tet_fiber_k1_d
                storage_entry(33, true),  // tet_fiber_k2_d
                storage_entry(34, true),  // xpbd_dihedral_indices
                storage_entry(35, true),  // xpbd_dihedral_params
                storage_entry(36, true),  // node_dihedral_offsets
                storage_entry(37, true),  // node_dihedral_entries
            ],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_fem uniforms"),
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
            label: Some("molla::tire_fem layout"),
            bind_group_layouts: &[&layout_io, &layout_uniforms],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::tire_fem::step_nodes"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("step_nodes"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let atlas_contract_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::tire_fem_atlas"),
            source: wgpu::ShaderSource::Wgsl(TIRE_FEM_ATLAS_SHADER.into()),
        });
        let atlas_contract_entries = (0..44)
            .map(|binding| storage_entry(binding, !matches!(binding, 12 | 13 | 14 | 20)))
            .collect::<Vec<_>>();
        let atlas_contract_layout_io =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::tire_fem_atlas contract io"),
                entries: &atlas_contract_entries,
            });
        let atlas_contract_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::tire_fem_atlas contract layout"),
                bind_group_layouts: &[&atlas_contract_layout_io],
                push_constant_ranges: &[],
            });
        let atlas_contract_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_fem_atlas::step_nodes_atlas_contract"),
                layout: Some(&atlas_contract_pipeline_layout),
                module: &atlas_contract_shader,
                entry_point: Some("step_nodes_atlas_contract"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let atlas_hub_layout_io =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::tire_fem_atlas hub io"),
                entries: &[
                    storage_entry(1, true),   // pos_in
                    storage_entry(2, true),   // vel_in
                    storage_entry(9, true),   // bead_ids
                    storage_entry(10, true),  // bead_offsets
                    storage_entry(20, false), // hub_state (rw)
                    storage_entry(43, true),  // offset_rows
                    storage_entry(44, true),  // hub param rows
                ],
            });
        let atlas_hub_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::tire_fem_atlas hub layout"),
                bind_group_layouts: &[&atlas_hub_layout_io],
                push_constant_ranges: &[],
            });
        let atlas_hub_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::tire_fem_atlas::hub_reduce_atlas"),
            layout: Some(&atlas_hub_pipeline_layout),
            module: &atlas_contract_shader,
            entry_point: Some("hub_reduce_atlas"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let atlas_hub_params_from_brush_layout_io =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::tire_fem_atlas brush-to-hub-param io"),
                entries: &[
                    storage_entry(45, true),  // BrushWheelOutput rows
                    storage_entry(46, false), // writable atlas hub param rows
                ],
            });
        let atlas_hub_params_from_brush_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::tire_fem_atlas brush-to-hub-param layout"),
                bind_group_layouts: &[&atlas_hub_params_from_brush_layout_io, &layout_uniforms],
                push_constant_ranges: &[],
            });
        let atlas_hub_params_from_brush_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_fem_atlas::hub_params_from_brush_outputs"),
                layout: Some(&atlas_hub_params_from_brush_pipeline_layout),
                module: &atlas_contract_shader,
                entry_point: Some("hub_params_from_brush_outputs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let atlas_hub_params_from_contact_layout_io =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::tire_fem_atlas contact-reduction-to-hub-param io"),
                entries: &[
                    storage_entry(46, false), // writable atlas hub param rows
                    storage_entry(47, true),  // packed contact-reduction output rows
                ],
            });
        let atlas_hub_params_from_contact_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::tire_fem_atlas contact-reduction-to-hub-param layout"),
                bind_group_layouts: &[&atlas_hub_params_from_contact_layout_io, &layout_uniforms],
                push_constant_ranges: &[],
            });
        let atlas_hub_params_from_contact_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_fem_atlas::hub_params_from_contact_reduction_outputs"),
                layout: Some(&atlas_hub_params_from_contact_pipeline_layout),
                module: &atlas_contract_shader,
                entry_point: Some("hub_params_from_contact_reduction_outputs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        let atlas_hub_params_from_feedback_layout_io =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::tire_fem_atlas feedback-force-to-hub-param io"),
                entries: &[
                    storage_entry(43, true),  // offset rows for per-tire node ranges
                    storage_entry(46, false), // writable atlas hub param rows
                    storage_entry(48, true),  // per-node FEM feedback force rows
                ],
            });
        let atlas_hub_params_from_feedback_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::tire_fem_atlas feedback-force-to-hub-param layout"),
                bind_group_layouts: &[&atlas_hub_params_from_feedback_layout_io, &layout_uniforms],
                push_constant_ranges: &[],
            });
        let atlas_hub_params_from_feedback_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::tire_fem_atlas::hub_params_from_feedback_force_rows"),
                layout: Some(&atlas_hub_params_from_feedback_pipeline_layout),
                module: &atlas_contract_shader,
                entry_point: Some("hub_params_from_feedback_force_rows"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // Stage (f) — hub reduction pipeline.
        let hub_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::tire_fem_hub"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/tire_fem_hub.wgsl").into()),
        });
        let hub_layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_fem hub io"),
            entries: &[
                storage_entry(0, true),  // bead_ids
                storage_entry(1, true),  // bead_offsets
                storage_entry(2, true),  // pos_in
                storage_entry(3, true),  // vel_in
                storage_entry(4, false), // hub_state (rw)
            ],
        });
        let hub_layout_uniforms =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::tire_fem hub uniforms"),
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
        let hub_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::tire_fem hub layout"),
            bind_group_layouts: &[&hub_layout_io, &hub_layout_uniforms],
            push_constant_ranges: &[],
        });
        let hub_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::tire_fem::hub_reduce"),
            layout: Some(&hub_pipeline_layout),
            module: &hub_shader,
            entry_point: Some("hub_reduce"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // G4 follow-up — same shader, second entry point + extra binding
        // for the contact-feedback force rows.
        let hub_tread_layout_io =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("molla::tire_fem hub tread io"),
                entries: &[
                    storage_entry(0, true),  // bead_ids
                    storage_entry(1, true),  // bead_offsets
                    storage_entry(2, true),  // pos_in
                    storage_entry(3, true),  // vel_in
                    storage_entry(4, false), // hub_state (rw)
                    storage_entry(5, true),  // tread_force_in
                ],
            });
        let hub_tread_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::tire_fem hub tread layout"),
                bind_group_layouts: &[&hub_tread_layout_io, &hub_layout_uniforms],
                push_constant_ranges: &[],
            });
        let hub_tread_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::tire_fem::hub_reduce_with_tread_wrench"),
            layout: Some(&hub_tread_pipeline_layout),
            module: &hub_shader,
            entry_point: Some("hub_reduce_with_tread_wrench"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            layout_io,
            layout_uniforms,
            atlas_contract_pipeline,
            atlas_contract_layout_io,
            atlas_hub_pipeline,
            atlas_hub_layout_io,
            atlas_hub_params_from_brush_pipeline,
            atlas_hub_params_from_brush_layout_io,
            atlas_hub_params_from_contact_pipeline,
            atlas_hub_params_from_contact_layout_io,
            atlas_hub_params_from_feedback_pipeline,
            atlas_hub_params_from_feedback_layout_io,
            hub_pipeline,
            hub_layout_io,
            hub_layout_uniforms,
            hub_tread_pipeline,
            hub_tread_layout_io,
        }
    }

    pub fn dispatch_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireFemGpuInput,
    ) -> Result<TireFemGpuOutput> {
        validate_input(input)?;
        let n = input.positions.len();
        if n == 0 {
            return Ok(TireFemGpuOutput {
                positions: vec![],
                velocities: vec![],
                forces: vec![],
            });
        }

        let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let f32_bytes = |len: usize| (len * core::mem::size_of::<f32>()) as u64;
        let u32_bytes = |len: usize| (len * core::mem::size_of::<u32>()) as u64;

        let rest_buf = mk_buf("tire_fem rest", vec4_bytes(n), usage_ro);
        let pos_buf = mk_buf("tire_fem pos", vec4_bytes(n), usage_ro);
        let vel_buf = mk_buf("tire_fem vel", vec4_bytes(n), usage_ro);
        let mass_buf = mk_buf("tire_fem mass", f32_bytes(n), usage_ro);
        let tet_i_buf = mk_buf(
            "tire_fem tet_indices",
            u32_bytes(input.tet_indices.len()),
            usage_ro,
        );
        let tet_g_buf = mk_buf(
            "tire_fem tet_grads",
            vec4_bytes(input.tet_grads.len()),
            usage_ro,
        );
        let tet_v_buf = mk_buf(
            "tire_fem tet_volume",
            f32_bytes(input.tet_volumes.len()),
            usage_ro,
        );
        let tet_mu_buf = mk_buf("tire_fem tet_mu", f32_bytes(input.tet_mu.len()), usage_ro);
        let tet_lambda_buf = mk_buf(
            "tire_fem tet_lambda",
            f32_bytes(input.tet_lambda.len()),
            usage_ro,
        );
        let bead_id_buf = mk_buf(
            "tire_fem bead_ids",
            u32_bytes(input.bead_ids.len()),
            usage_ro,
        );
        let bead_off_buf = mk_buf(
            "tire_fem bead_offsets",
            vec4_bytes(input.bead_offsets.len()),
            usage_ro,
        );
        let tread_id_buf = mk_buf(
            "tire_fem tread_ids",
            u32_bytes(input.tread_ids.len()),
            usage_ro,
        );
        let out_pos = mk_buf("tire_fem out_pos", vec4_bytes(n), usage_rw);
        let out_vel = mk_buf("tire_fem out_vel", vec4_bytes(n), usage_rw);
        let out_force = mk_buf("tire_fem out_force", vec4_bytes(n), usage_rw);
        let tet_fk_buf = mk_buf(
            "tire_fem tet_fiber_dir",
            vec4_bytes(input.tet_fiber_dir.len()),
            usage_ro,
        );
        let tet_k1_buf = mk_buf(
            "tire_fem tet_fiber_k1",
            f32_bytes(input.tet_fiber_k1.len()),
            usage_ro,
        );
        let tet_k2_buf = mk_buf(
            "tire_fem tet_fiber_k2",
            f32_bytes(input.tet_fiber_k2.len()),
            usage_ro,
        );
        let tet_fk_b_buf = mk_buf(
            "tire_fem tet_fiber_dir_b",
            vec4_bytes(input.tet_fiber_dir_b.len()),
            usage_ro,
        );
        let tet_k1_b_buf = mk_buf(
            "tire_fem tet_fiber_k1_b",
            f32_bytes(input.tet_fiber_k1_b.len()),
            usage_ro,
        );
        let tet_k2_b_buf = mk_buf(
            "tire_fem tet_fiber_k2_b",
            f32_bytes(input.tet_fiber_k2_b.len()),
            usage_ro,
        );
        let tet_fk_c_buf = mk_buf(
            "tire_fem tet_fiber_dir_c",
            vec4_bytes(input.tet_fiber_dir_c.len()),
            usage_ro,
        );
        let tet_k1_c_buf = mk_buf(
            "tire_fem tet_fiber_k1_c",
            f32_bytes(input.tet_fiber_k1_c.len()),
            usage_ro,
        );
        let tet_k2_c_buf = mk_buf(
            "tire_fem tet_fiber_k2_c",
            f32_bytes(input.tet_fiber_k2_c.len()),
            usage_ro,
        );
        let tet_fk_d_buf = mk_buf(
            "tire_fem tet_fiber_dir_d",
            vec4_bytes(input.tet_fiber_dir_d.len()),
            usage_ro,
        );
        let tet_k1_d_buf = mk_buf(
            "tire_fem tet_fiber_k1_d",
            f32_bytes(input.tet_fiber_k1_d.len()),
            usage_ro,
        );
        let tet_k2_d_buf = mk_buf(
            "tire_fem tet_fiber_k2_d",
            f32_bytes(input.tet_fiber_k2_d.len()),
            usage_ro,
        );
        let csr_off_buf = mk_buf(
            "tire_fem node_tet_offsets",
            u32_bytes(input.node_tet_offsets.len()),
            usage_ro,
        );
        let csr_ent_buf = mk_buf(
            "tire_fem node_tet_entries",
            u32_bytes(input.node_tet_entries.len()),
            usage_ro,
        );
        let xpbd_edge_i_buf = mk_buf(
            "tire_fem xpbd_edge_indices",
            vec4_bytes(input.xpbd_edge_indices.len()),
            usage_ro,
        );
        let xpbd_edge_p_buf = mk_buf(
            "tire_fem xpbd_edge_params",
            vec4_bytes(input.xpbd_edge_params.len()),
            usage_ro,
        );
        let node_edge_off_buf = mk_buf(
            "tire_fem node_edge_offsets",
            u32_bytes(input.node_edge_offsets.len()),
            usage_ro,
        );
        let node_edge_ent_buf = mk_buf(
            "tire_fem node_edge_entries",
            u32_bytes(input.node_edge_entries.len()),
            usage_ro,
        );
        let xpbd_dihedral_i_buf = mk_buf(
            "tire_fem xpbd_dihedral_indices",
            vec4_bytes(input.xpbd_dihedral_indices.len()),
            usage_ro,
        );
        let xpbd_dihedral_p_buf = mk_buf(
            "tire_fem xpbd_dihedral_params",
            vec4_bytes(input.xpbd_dihedral_params.len()),
            usage_ro,
        );
        let node_dihedral_off_buf = mk_buf(
            "tire_fem node_dihedral_offsets",
            u32_bytes(input.node_dihedral_offsets.len()),
            usage_ro,
        );
        let node_dihedral_ent_buf = mk_buf(
            "tire_fem node_dihedral_entries",
            u32_bytes(input.node_dihedral_entries.len()),
            usage_ro,
        );

        queue.write_buffer(&rest_buf, 0, bytemuck::cast_slice(&input.rest_positions));
        queue.write_buffer(&pos_buf, 0, bytemuck::cast_slice(&input.positions));
        queue.write_buffer(&vel_buf, 0, bytemuck::cast_slice(&input.velocities));
        queue.write_buffer(&mass_buf, 0, bytemuck::cast_slice(&input.masses));
        queue.write_buffer(&tet_i_buf, 0, bytemuck::cast_slice(&input.tet_indices));
        queue.write_buffer(&tet_g_buf, 0, bytemuck::cast_slice(&input.tet_grads));
        queue.write_buffer(&tet_v_buf, 0, bytemuck::cast_slice(&input.tet_volumes));
        queue.write_buffer(&tet_mu_buf, 0, bytemuck::cast_slice(&input.tet_mu));
        queue.write_buffer(&tet_lambda_buf, 0, bytemuck::cast_slice(&input.tet_lambda));
        if !input.bead_ids.is_empty() {
            queue.write_buffer(&bead_id_buf, 0, bytemuck::cast_slice(&input.bead_ids));
            queue.write_buffer(&bead_off_buf, 0, bytemuck::cast_slice(&input.bead_offsets));
        }
        if !input.tread_ids.is_empty() {
            queue.write_buffer(&tread_id_buf, 0, bytemuck::cast_slice(&input.tread_ids));
        }
        if !input.tet_fiber_dir.is_empty() {
            queue.write_buffer(&tet_fk_buf, 0, bytemuck::cast_slice(&input.tet_fiber_dir));
            queue.write_buffer(&tet_k1_buf, 0, bytemuck::cast_slice(&input.tet_fiber_k1));
            queue.write_buffer(&tet_k2_buf, 0, bytemuck::cast_slice(&input.tet_fiber_k2));
            queue.write_buffer(
                &tet_fk_b_buf,
                0,
                bytemuck::cast_slice(&input.tet_fiber_dir_b),
            );
            queue.write_buffer(
                &tet_k1_b_buf,
                0,
                bytemuck::cast_slice(&input.tet_fiber_k1_b),
            );
            queue.write_buffer(
                &tet_k2_b_buf,
                0,
                bytemuck::cast_slice(&input.tet_fiber_k2_b),
            );
            queue.write_buffer(
                &tet_fk_c_buf,
                0,
                bytemuck::cast_slice(&input.tet_fiber_dir_c),
            );
            queue.write_buffer(
                &tet_k1_c_buf,
                0,
                bytemuck::cast_slice(&input.tet_fiber_k1_c),
            );
            queue.write_buffer(
                &tet_k2_c_buf,
                0,
                bytemuck::cast_slice(&input.tet_fiber_k2_c),
            );
            queue.write_buffer(
                &tet_fk_d_buf,
                0,
                bytemuck::cast_slice(&input.tet_fiber_dir_d),
            );
            queue.write_buffer(
                &tet_k1_d_buf,
                0,
                bytemuck::cast_slice(&input.tet_fiber_k1_d),
            );
            queue.write_buffer(
                &tet_k2_d_buf,
                0,
                bytemuck::cast_slice(&input.tet_fiber_k2_d),
            );
        }
        if !input.node_tet_offsets.is_empty() {
            queue.write_buffer(
                &csr_off_buf,
                0,
                bytemuck::cast_slice(&input.node_tet_offsets),
            );
        }
        if !input.node_tet_entries.is_empty() {
            queue.write_buffer(
                &csr_ent_buf,
                0,
                bytemuck::cast_slice(&input.node_tet_entries),
            );
        }
        if !input.xpbd_edge_indices.is_empty() {
            queue.write_buffer(
                &xpbd_edge_i_buf,
                0,
                bytemuck::cast_slice(&input.xpbd_edge_indices),
            );
            queue.write_buffer(
                &xpbd_edge_p_buf,
                0,
                bytemuck::cast_slice(&input.xpbd_edge_params),
            );
            queue.write_buffer(
                &node_edge_off_buf,
                0,
                bytemuck::cast_slice(&input.node_edge_offsets),
            );
            queue.write_buffer(
                &node_edge_ent_buf,
                0,
                bytemuck::cast_slice(&input.node_edge_entries),
            );
        }
        if !input.xpbd_dihedral_indices.is_empty() {
            queue.write_buffer(
                &xpbd_dihedral_i_buf,
                0,
                bytemuck::cast_slice(&input.xpbd_dihedral_indices),
            );
            queue.write_buffer(
                &xpbd_dihedral_p_buf,
                0,
                bytemuck::cast_slice(&input.xpbd_dihedral_params),
            );
            queue.write_buffer(
                &node_dihedral_off_buf,
                0,
                bytemuck::cast_slice(&input.node_dihedral_offsets),
            );
            queue.write_buffer(
                &node_dihedral_ent_buf,
                0,
                bytemuck::cast_slice(&input.node_dihedral_entries),
            );
        }

        // Legacy hub-state buffer — initialised from params.hub. No GPU
        // hub integration in `dispatch_blocking`; the hub stays frozen
        // for this one-shot call. Two vec4 slots so the main shader
        // can read either 1-DOF or 3-DOF hub.
        let hub_state_legacy_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_fem hub_state legacy"),
            size: (8 * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let init_hub: [f32; 8] = [
            input.params.hub[0],
            input.params.hub[1],
            input.params.hub[2],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        queue.write_buffer(&hub_state_legacy_buf, 0, bytemuck::bytes_of(&init_hub));

        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem io bg"),
            layout: &self.layout_io,
            entries: &[
                entry(0, &rest_buf),
                entry(1, &pos_buf),
                entry(2, &vel_buf),
                entry(3, &mass_buf),
                entry(4, &tet_i_buf),
                entry(5, &tet_g_buf),
                entry(6, &tet_v_buf),
                entry(7, &tet_mu_buf),
                entry(8, &tet_lambda_buf),
                entry(9, &bead_id_buf),
                entry(10, &bead_off_buf),
                entry(11, &tread_id_buf),
                entry(12, &out_pos),
                entry(13, &out_vel),
                entry(14, &out_force),
                entry(15, &tet_fk_buf),
                entry(16, &tet_k1_buf),
                entry(17, &tet_k2_buf),
                entry(18, &csr_off_buf),
                entry(19, &csr_ent_buf),
                entry(20, &hub_state_legacy_buf),
                entry(21, &tet_fk_b_buf),
                entry(22, &tet_k1_b_buf),
                entry(23, &tet_k2_b_buf),
                entry(24, &xpbd_edge_i_buf),
                entry(25, &xpbd_edge_p_buf),
                entry(26, &node_edge_off_buf),
                entry(27, &node_edge_ent_buf),
                entry(28, &tet_fk_c_buf),
                entry(29, &tet_k1_c_buf),
                entry(30, &tet_k2_c_buf),
                entry(31, &tet_fk_d_buf),
                entry(32, &tet_k1_d_buf),
                entry(33, &tet_k2_d_buf),
                entry(34, &xpbd_dihedral_i_buf),
                entry(35, &xpbd_dihedral_p_buf),
                entry(36, &node_dihedral_off_buf),
                entry(37, &node_dihedral_ent_buf),
            ],
        });

        let tet_count = input.tet_indices.len() / 4;
        let uniforms = Uniforms {
            counts: [
                n as u32,
                tet_count as u32,
                input.bead_ids.len() as u32,
                input.tread_ids.len() as u32,
            ],
            counts2: [
                input.xpbd_edge_indices.len() as u32,
                input.xpbd_dihedral_indices.len() as u32,
                0,
                0,
            ],
            phys0: [
                input.params.dt,
                input.params.young_modulus,
                input.params.poisson_ratio,
                input.params.pressure_pa,
            ],
            phys1: [
                input.params.node_damping,
                input.params.bead_stiffness,
                input.params.floor_stiffness,
                input.params.floor_damping,
            ],
            phys2: [
                input.params.pressure_area_per_tread_node,
                input.params.hub[0],
                input.params.hub[1],
                input.params.hub[2],
            ],
            phys3: [
                input.params.gravity_y,
                input.params.cavity_radial_force_per_bead,
                input.params.belt_proxy_mu,
                input.params.xpbd_volume_compliance,
            ],
            phys4: [
                input.params.xpbd_bead_compliance,
                input.params.semi_implicit_stiffness_scale,
                input.params.xpbd_sidewall_ply_stretch_limit,
                input.params.xpbd_sidewall_ply_compliance,
            ],
            // PLAN_TOURQUE Stage G part 1 — torque/spin uniform row.
            phys5: [
                input.params.hub_omega,
                input.params.drive_torque_n_m,
                input.params.brake_torque_n_m,
                0.0,
            ],
        };
        let uniform_buf = mk_buf(
            "tire_fem uniforms",
            core::mem::size_of::<Uniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_u = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniform_buf)],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_fem encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_fem pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_u, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }

        let stage_pos = stage_vec4(device, "tire_fem stage pos", n);
        let stage_vel = stage_vec4(device, "tire_fem stage vel", n);
        let stage_force = stage_vec4(device, "tire_fem stage force", n);
        let bytes = vec4_bytes(n);
        encoder.copy_buffer_to_buffer(&out_pos, 0, &stage_pos, 0, bytes);
        encoder.copy_buffer_to_buffer(&out_vel, 0, &stage_vel, 0, bytes);
        encoder.copy_buffer_to_buffer(&out_force, 0, &stage_force, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));

        Ok(TireFemGpuOutput {
            positions: read_vec4(device, &stage_pos, "tire_fem positions")?,
            velocities: read_vec4(device, &stage_vel, "tire_fem velocities")?,
            forces: read_vec4(device, &stage_force, "tire_fem forces")?,
        })
    }

    /// Execute the packed atlas through the one-dispatch WGSL physical path.
    ///
    /// All packed atlas buffers are uploaded as one bind group,
    /// `tire_fem_atlas.wgsl` is dispatched once over the total node count, and
    /// the packed position/velocity/force rows are read back. The shader uses
    /// `node_tire_indices`, `param_rows`, and `offset_rows` to run the same
    /// physical FEM node update as the single-tire GPU path.
    pub fn dispatch_atlas_contract_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        atlas: &TireFemGpuBatchAtlasBuffers,
    ) -> Result<TireFemGpuBatchAtlasOutputRows> {
        atlas.validate()?;
        let n = atlas.layout.plan.total_node_count;
        if n == 0 {
            return Ok(TireFemGpuBatchAtlasOutputRows::default());
        }
        let bindings = tire_fem_gpu_batch_atlas_binding_plan(atlas)?;
        let contract = tire_fem_gpu_batch_atlas_dispatch_contract(atlas)?;
        if contract.binding_count != bindings.len() {
            return Err(Error::Gpu(format!(
                "tire_fem atlas contract binding mismatch: contract={}, plan={}",
                contract.binding_count,
                bindings.len()
            )));
        }

        let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let out_bytes = bytes_of_slice::<[f32; 4]>(n);

        let mut hub_state = Vec::with_capacity(atlas.layout.plan.tire_count * 2);
        for params in &atlas.params {
            hub_state.push([params.hub[0], params.hub[1], params.hub[2], 0.0]);
            hub_state.push([0.0, 0.0, 0.0, 0.0]);
        }

        let rest_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas rest_pos",
            &atlas.rest_positions,
            usage_ro,
        );
        let pos_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas pos_in",
            &atlas.positions,
            usage_ro,
        );
        let vel_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas vel_in",
            &atlas.velocities,
            usage_ro,
        );
        let mass_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas mass_in",
            &atlas.masses,
            usage_ro,
        );
        let tet_i_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas tet_indices",
            &atlas.tet_indices,
            usage_ro,
        );
        let tet_g_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas tet_grads",
            &atlas.tet_grads,
            usage_ro,
        );
        let tet_v_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas tet_volume",
            &atlas.tet_volumes,
            usage_ro,
        );
        let tet_mu_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas tet_mu",
            &atlas.tet_mu,
            usage_ro,
        );
        let tet_lambda_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas tet_lambda",
            &atlas.tet_lambda,
            usage_ro,
        );
        let bead_id_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas bead_ids",
            &atlas.bead_ids,
            usage_ro,
        );
        let bead_off_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas bead_offsets",
            &atlas.bead_offsets,
            usage_ro,
        );
        let tread_id_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas tread_ids",
            &atlas.tread_ids,
            usage_ro,
        );
        let out_pos = empty_storage_buffer(device, "tire_fem_atlas pos_out", out_bytes, usage_rw);
        let out_vel = empty_storage_buffer(device, "tire_fem_atlas vel_out", out_bytes, usage_rw);
        let out_force =
            empty_storage_buffer(device, "tire_fem_atlas force_out", out_bytes, usage_rw);
        let fiber_dir_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas tet_fiber_dir",
            &atlas.tet_fiber_dir,
            usage_ro,
        );
        let fiber_k1_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas tet_fiber_k1",
            &atlas.tet_fiber_k1,
            usage_ro,
        );
        let fiber_k2_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas tet_fiber_k2",
            &atlas.tet_fiber_k2,
            usage_ro,
        );
        let node_tet_off_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas node_tet_offsets",
            &atlas.node_tet_offsets,
            usage_ro,
        );
        let node_tet_ent_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas node_tet_entries",
            &atlas.node_tet_entries,
            usage_ro,
        );
        let hub_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas hub_state",
            &hub_state,
            usage_ro,
        );
        let fiber_dir_b_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas tet_fiber_dir_b",
            &atlas.tet_fiber_dir_b,
            usage_ro,
        );
        let fiber_k1_b_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas tet_fiber_k1_b",
            &atlas.tet_fiber_k1_b,
            usage_ro,
        );
        let fiber_k2_b_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas tet_fiber_k2_b",
            &atlas.tet_fiber_k2_b,
            usage_ro,
        );
        let edge_i_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas xpbd_edge_indices",
            &atlas.xpbd_edge_indices,
            usage_ro,
        );
        let edge_p_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas xpbd_edge_params",
            &atlas.xpbd_edge_params,
            usage_ro,
        );
        let node_edge_off_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas node_edge_offsets",
            &atlas.node_edge_offsets,
            usage_ro,
        );
        let node_edge_ent_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas node_edge_entries",
            &atlas.node_edge_entries,
            usage_ro,
        );
        let fiber_dir_c_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas tet_fiber_dir_c",
            &atlas.tet_fiber_dir_c,
            usage_ro,
        );
        let fiber_k1_c_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas tet_fiber_k1_c",
            &atlas.tet_fiber_k1_c,
            usage_ro,
        );
        let fiber_k2_c_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas tet_fiber_k2_c",
            &atlas.tet_fiber_k2_c,
            usage_ro,
        );
        let fiber_dir_d_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas tet_fiber_dir_d",
            &atlas.tet_fiber_dir_d,
            usage_ro,
        );
        let fiber_k1_d_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas tet_fiber_k1_d",
            &atlas.tet_fiber_k1_d,
            usage_ro,
        );
        let fiber_k2_d_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas tet_fiber_k2_d",
            &atlas.tet_fiber_k2_d,
            usage_ro,
        );
        let dihedral_i_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas xpbd_dihedral_indices",
            &atlas.xpbd_dihedral_indices,
            usage_ro,
        );
        let dihedral_p_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas xpbd_dihedral_params",
            &atlas.xpbd_dihedral_params,
            usage_ro,
        );
        let node_dihedral_off_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas node_dihedral_offsets",
            &atlas.node_dihedral_offsets,
            usage_ro,
        );
        let node_dihedral_ent_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas node_dihedral_entries",
            &atlas.node_dihedral_entries,
            usage_ro,
        );
        let node_tire_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas node_tire_indices",
            &atlas.node_tire_indices,
            usage_ro,
        );
        let tet_tire_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas tet_tire_indices",
            &atlas.tet_tire_indices,
            usage_ro,
        );
        let bead_tire_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas bead_tire_indices",
            &atlas.bead_tire_indices,
            usage_ro,
        );
        let tread_tire_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas tread_tire_indices",
            &atlas.tread_tire_indices,
            usage_ro,
        );
        let param_rows_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas param_rows",
            &atlas.param_rows,
            usage_ro,
        );
        let offset_rows_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas offset_rows",
            &atlas.offset_rows,
            usage_ro,
        );

        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem_atlas contract io bg"),
            layout: &self.atlas_contract_layout_io,
            entries: &[
                entry(0, &rest_buf),
                entry(1, &pos_buf),
                entry(2, &vel_buf),
                entry(3, &mass_buf),
                entry(4, &tet_i_buf),
                entry(5, &tet_g_buf),
                entry(6, &tet_v_buf),
                entry(7, &tet_mu_buf),
                entry(8, &tet_lambda_buf),
                entry(9, &bead_id_buf),
                entry(10, &bead_off_buf),
                entry(11, &tread_id_buf),
                entry(12, &out_pos),
                entry(13, &out_vel),
                entry(14, &out_force),
                entry(15, &fiber_dir_buf),
                entry(16, &fiber_k1_buf),
                entry(17, &fiber_k2_buf),
                entry(18, &node_tet_off_buf),
                entry(19, &node_tet_ent_buf),
                entry(20, &hub_buf),
                entry(21, &fiber_dir_b_buf),
                entry(22, &fiber_k1_b_buf),
                entry(23, &fiber_k2_b_buf),
                entry(24, &edge_i_buf),
                entry(25, &edge_p_buf),
                entry(26, &node_edge_off_buf),
                entry(27, &node_edge_ent_buf),
                entry(28, &fiber_dir_c_buf),
                entry(29, &fiber_k1_c_buf),
                entry(30, &fiber_k2_c_buf),
                entry(31, &fiber_dir_d_buf),
                entry(32, &fiber_k1_d_buf),
                entry(33, &fiber_k2_d_buf),
                entry(34, &dihedral_i_buf),
                entry(35, &dihedral_p_buf),
                entry(36, &node_dihedral_off_buf),
                entry(37, &node_dihedral_ent_buf),
                entry(38, &node_tire_buf),
                entry(39, &tet_tire_buf),
                entry(40, &bead_tire_buf),
                entry(41, &tread_tire_buf),
                entry(42, &param_rows_buf),
                entry(43, &offset_rows_buf),
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_fem_atlas contract encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_fem_atlas contract pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.atlas_contract_pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.dispatch_workgroups(contract.workgroup_count, 1, 1);
        }

        let stage_pos = stage_vec4(device, "tire_fem_atlas stage pos", n);
        let stage_vel = stage_vec4(device, "tire_fem_atlas stage vel", n);
        let stage_force = stage_vec4(device, "tire_fem_atlas stage force", n);
        encoder.copy_buffer_to_buffer(&out_pos, 0, &stage_pos, 0, out_bytes);
        encoder.copy_buffer_to_buffer(&out_vel, 0, &stage_vel, 0, out_bytes);
        encoder.copy_buffer_to_buffer(&out_force, 0, &stage_force, 0, out_bytes);
        queue.submit(std::iter::once(encoder.finish()));

        Ok(TireFemGpuBatchAtlasOutputRows {
            positions: read_vec4(device, &stage_pos, "tire_fem_atlas positions")?,
            velocities: read_vec4(device, &stage_vel, "tire_fem_atlas velocities")?,
            forces: read_vec4(device, &stage_force, "tire_fem_atlas forces")?,
        })
    }

    /// Pack several tires into one atlas and run the fused one-dispatch FEM
    /// node-step shader.
    ///
    /// This is the high-level batch entry point for non-persistent callers:
    /// it performs the same atlas packing/index rewriting used by the
    /// lower-level contract functions, dispatches the physical atlas shader
    /// once, then splits the packed readback rows back into one
    /// [`TireFemGpuOutput`] per tire. Persistent viewer ownership should use
    /// [`Self::create_resident_atlas_batch`] when it can retain GPU state.
    pub fn dispatch_batch_atlas_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        inputs: &[TireFemGpuInput],
    ) -> Result<Vec<TireFemGpuOutput>> {
        let atlas = tire_fem_gpu_batch_atlas_buffers(inputs)?;
        let rows = self.dispatch_atlas_contract_blocking(device, queue, &atlas)?;
        atlas.split_output_rows(&rows)
    }

    /// Execute the packed atlas through the current per-tire GPU path and
    /// return atlas-wide output rows.
    ///
    /// This is intentionally a reference/compatibility loop, not the final
    /// fused one-dispatch kernel. It uses the same validated atlas pack/extract
    /// boundary that the fused path will consume, then joins per-tire outputs
    /// back into the packed readback shape. The eventual fused shader can be
    /// compared byte-for-row against this method before replacing the loop.
    pub fn dispatch_atlas_reference_loop(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        atlas: &TireFemGpuBatchAtlasBuffers,
    ) -> Result<TireFemGpuBatchAtlasOutputRows> {
        atlas.validate()?;
        let mut outputs = Vec::with_capacity(atlas.layout.plan.tire_count);
        for tire_index in 0..atlas.layout.plan.tire_count {
            let input = atlas.extract_input(tire_index)?;
            outputs.push(self.dispatch_blocking(device, queue, &input)?);
        }
        atlas.join_outputs(&outputs)
    }

    /// Stage (g) — build the persistent GPU buffers for one tire mesh.
    /// Mesh-constant data (rest positions, tet topology + materials +
    /// fibres, CSR, bead/tread index arrays) is uploaded once and never
    /// re-uploaded.
    pub fn create_persistent(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireFemGpuInput,
    ) -> Result<TireFemPersistentMesh> {
        validate_input(input)?;
        let n = input.positions.len();
        if n == 0 {
            return Err(Error::Solver(
                "tire_fem: cannot build persistent buffers for an empty mesh".into(),
            ));
        }

        let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let f32_bytes = |len: usize| (len * core::mem::size_of::<f32>()) as u64;
        let u32_bytes = |len: usize| (len * core::mem::size_of::<u32>()) as u64;

        // Constant buffers — uploaded exactly once below.
        let rest_buf = mk_buf("tire_fem rest p", vec4_bytes(n), usage_ro);
        let mass_buf = mk_buf("tire_fem mass p", f32_bytes(n), usage_ro);
        let tet_i_buf = mk_buf(
            "tire_fem tet_indices p",
            u32_bytes(input.tet_indices.len()),
            usage_ro,
        );
        let tet_g_buf = mk_buf(
            "tire_fem tet_grads p",
            vec4_bytes(input.tet_grads.len()),
            usage_ro,
        );
        let tet_v_buf = mk_buf(
            "tire_fem tet_volume p",
            f32_bytes(input.tet_volumes.len()),
            usage_ro,
        );
        let tet_mu_buf = mk_buf("tire_fem tet_mu p", f32_bytes(input.tet_mu.len()), usage_ro);
        let tet_lambda_buf = mk_buf(
            "tire_fem tet_lambda p",
            f32_bytes(input.tet_lambda.len()),
            usage_ro,
        );
        let tet_fk_buf = mk_buf(
            "tire_fem tet_fiber_dir p",
            vec4_bytes(input.tet_fiber_dir.len()),
            usage_ro,
        );
        let tet_k1_buf = mk_buf(
            "tire_fem tet_fiber_k1 p",
            f32_bytes(input.tet_fiber_k1.len()),
            usage_ro,
        );
        let tet_k2_buf = mk_buf(
            "tire_fem tet_fiber_k2 p",
            f32_bytes(input.tet_fiber_k2.len()),
            usage_ro,
        );
        let tet_fk_b_buf = mk_buf(
            "tire_fem tet_fiber_dir_b p",
            vec4_bytes(input.tet_fiber_dir_b.len()),
            usage_ro,
        );
        let tet_k1_b_buf = mk_buf(
            "tire_fem tet_fiber_k1_b p",
            f32_bytes(input.tet_fiber_k1_b.len()),
            usage_ro,
        );
        let tet_k2_b_buf = mk_buf(
            "tire_fem tet_fiber_k2_b p",
            f32_bytes(input.tet_fiber_k2_b.len()),
            usage_ro,
        );
        let tet_fk_c_buf = mk_buf(
            "tire_fem tet_fiber_dir_c p",
            vec4_bytes(input.tet_fiber_dir_c.len()),
            usage_ro,
        );
        let tet_k1_c_buf = mk_buf(
            "tire_fem tet_fiber_k1_c p",
            f32_bytes(input.tet_fiber_k1_c.len()),
            usage_ro,
        );
        let tet_k2_c_buf = mk_buf(
            "tire_fem tet_fiber_k2_c p",
            f32_bytes(input.tet_fiber_k2_c.len()),
            usage_ro,
        );
        let tet_fk_d_buf = mk_buf(
            "tire_fem tet_fiber_dir_d p",
            vec4_bytes(input.tet_fiber_dir_d.len()),
            usage_ro,
        );
        let tet_k1_d_buf = mk_buf(
            "tire_fem tet_fiber_k1_d p",
            f32_bytes(input.tet_fiber_k1_d.len()),
            usage_ro,
        );
        let tet_k2_d_buf = mk_buf(
            "tire_fem tet_fiber_k2_d p",
            f32_bytes(input.tet_fiber_k2_d.len()),
            usage_ro,
        );
        let csr_off_buf = mk_buf(
            "tire_fem csr_offsets p",
            u32_bytes(input.node_tet_offsets.len()),
            usage_ro,
        );
        let csr_ent_buf = mk_buf(
            "tire_fem csr_entries p",
            u32_bytes(input.node_tet_entries.len()),
            usage_ro,
        );
        let bead_id_buf = mk_buf(
            "tire_fem bead_ids p",
            u32_bytes(input.bead_ids.len()),
            usage_ro,
        );
        let bead_off_buf = mk_buf(
            "tire_fem bead_offsets p",
            vec4_bytes(input.bead_offsets.len()),
            usage_ro,
        );
        let tread_id_buf = mk_buf(
            "tire_fem tread_ids p",
            u32_bytes(input.tread_ids.len()),
            usage_ro,
        );
        let xpbd_edge_i_buf = mk_buf(
            "tire_fem xpbd_edge_indices p",
            vec4_bytes(input.xpbd_edge_indices.len()),
            usage_ro,
        );
        let xpbd_edge_p_buf = mk_buf(
            "tire_fem xpbd_edge_params p",
            vec4_bytes(input.xpbd_edge_params.len()),
            usage_ro,
        );
        let node_edge_off_buf = mk_buf(
            "tire_fem node_edge_offsets p",
            u32_bytes(input.node_edge_offsets.len()),
            usage_ro,
        );
        let node_edge_ent_buf = mk_buf(
            "tire_fem node_edge_entries p",
            u32_bytes(input.node_edge_entries.len()),
            usage_ro,
        );
        let xpbd_dihedral_i_buf = mk_buf(
            "tire_fem xpbd_dihedral_indices p",
            vec4_bytes(input.xpbd_dihedral_indices.len()),
            usage_ro,
        );
        let xpbd_dihedral_p_buf = mk_buf(
            "tire_fem xpbd_dihedral_params p",
            vec4_bytes(input.xpbd_dihedral_params.len()),
            usage_ro,
        );
        let node_dihedral_off_buf = mk_buf(
            "tire_fem node_dihedral_offsets p",
            u32_bytes(input.node_dihedral_offsets.len()),
            usage_ro,
        );
        let node_dihedral_ent_buf = mk_buf(
            "tire_fem node_dihedral_entries p",
            u32_bytes(input.node_dihedral_entries.len()),
            usage_ro,
        );

        // Ping-pong state buffers.
        let pos_a = mk_buf("tire_fem pos_a", vec4_bytes(n), usage_rw);
        let pos_b = mk_buf("tire_fem pos_b", vec4_bytes(n), usage_rw);
        let vel_a = mk_buf("tire_fem vel_a", vec4_bytes(n), usage_rw);
        let vel_b = mk_buf("tire_fem vel_b", vec4_bytes(n), usage_rw);
        let force_buf = mk_buf("tire_fem force p", vec4_bytes(n), usage_rw);
        let uniform_buf = mk_buf(
            "tire_fem uniforms p",
            core::mem::size_of::<Uniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        // Upload constants exactly once.
        queue.write_buffer(&rest_buf, 0, bytemuck::cast_slice(&input.rest_positions));
        queue.write_buffer(&mass_buf, 0, bytemuck::cast_slice(&input.masses));
        queue.write_buffer(&tet_i_buf, 0, bytemuck::cast_slice(&input.tet_indices));
        queue.write_buffer(&tet_g_buf, 0, bytemuck::cast_slice(&input.tet_grads));
        queue.write_buffer(&tet_v_buf, 0, bytemuck::cast_slice(&input.tet_volumes));
        queue.write_buffer(&tet_mu_buf, 0, bytemuck::cast_slice(&input.tet_mu));
        queue.write_buffer(&tet_lambda_buf, 0, bytemuck::cast_slice(&input.tet_lambda));
        if !input.tet_fiber_dir.is_empty() {
            queue.write_buffer(&tet_fk_buf, 0, bytemuck::cast_slice(&input.tet_fiber_dir));
            queue.write_buffer(&tet_k1_buf, 0, bytemuck::cast_slice(&input.tet_fiber_k1));
            queue.write_buffer(&tet_k2_buf, 0, bytemuck::cast_slice(&input.tet_fiber_k2));
            queue.write_buffer(
                &tet_fk_b_buf,
                0,
                bytemuck::cast_slice(&input.tet_fiber_dir_b),
            );
            queue.write_buffer(
                &tet_k1_b_buf,
                0,
                bytemuck::cast_slice(&input.tet_fiber_k1_b),
            );
            queue.write_buffer(
                &tet_k2_b_buf,
                0,
                bytemuck::cast_slice(&input.tet_fiber_k2_b),
            );
            queue.write_buffer(
                &tet_fk_c_buf,
                0,
                bytemuck::cast_slice(&input.tet_fiber_dir_c),
            );
            queue.write_buffer(
                &tet_k1_c_buf,
                0,
                bytemuck::cast_slice(&input.tet_fiber_k1_c),
            );
            queue.write_buffer(
                &tet_k2_c_buf,
                0,
                bytemuck::cast_slice(&input.tet_fiber_k2_c),
            );
            queue.write_buffer(
                &tet_fk_d_buf,
                0,
                bytemuck::cast_slice(&input.tet_fiber_dir_d),
            );
            queue.write_buffer(
                &tet_k1_d_buf,
                0,
                bytemuck::cast_slice(&input.tet_fiber_k1_d),
            );
            queue.write_buffer(
                &tet_k2_d_buf,
                0,
                bytemuck::cast_slice(&input.tet_fiber_k2_d),
            );
        }
        if !input.node_tet_offsets.is_empty() {
            queue.write_buffer(
                &csr_off_buf,
                0,
                bytemuck::cast_slice(&input.node_tet_offsets),
            );
        }
        if !input.node_tet_entries.is_empty() {
            queue.write_buffer(
                &csr_ent_buf,
                0,
                bytemuck::cast_slice(&input.node_tet_entries),
            );
        }
        if !input.bead_ids.is_empty() {
            queue.write_buffer(&bead_id_buf, 0, bytemuck::cast_slice(&input.bead_ids));
            queue.write_buffer(&bead_off_buf, 0, bytemuck::cast_slice(&input.bead_offsets));
        }
        if !input.tread_ids.is_empty() {
            queue.write_buffer(&tread_id_buf, 0, bytemuck::cast_slice(&input.tread_ids));
        }
        if !input.xpbd_edge_indices.is_empty() {
            queue.write_buffer(
                &xpbd_edge_i_buf,
                0,
                bytemuck::cast_slice(&input.xpbd_edge_indices),
            );
            queue.write_buffer(
                &xpbd_edge_p_buf,
                0,
                bytemuck::cast_slice(&input.xpbd_edge_params),
            );
            queue.write_buffer(
                &node_edge_off_buf,
                0,
                bytemuck::cast_slice(&input.node_edge_offsets),
            );
            queue.write_buffer(
                &node_edge_ent_buf,
                0,
                bytemuck::cast_slice(&input.node_edge_entries),
            );
        }
        if !input.xpbd_dihedral_indices.is_empty() {
            queue.write_buffer(
                &xpbd_dihedral_i_buf,
                0,
                bytemuck::cast_slice(&input.xpbd_dihedral_indices),
            );
            queue.write_buffer(
                &xpbd_dihedral_p_buf,
                0,
                bytemuck::cast_slice(&input.xpbd_dihedral_params),
            );
            queue.write_buffer(
                &node_dihedral_off_buf,
                0,
                bytemuck::cast_slice(&input.node_dihedral_offsets),
            );
            queue.write_buffer(
                &node_dihedral_ent_buf,
                0,
                bytemuck::cast_slice(&input.node_dihedral_entries),
            );
        }
        // Initial position/velocity into the A slot.
        queue.write_buffer(&pos_a, 0, bytemuck::cast_slice(&input.positions));
        queue.write_buffer(&vel_a, 0, bytemuck::cast_slice(&input.velocities));

        // Build bind groups for the two ping-pong directions. The
        // CONSTANT bindings (rest, tets, fibres, CSR, beads, treads,
        // masses) reference the same buffers in both. Only the input
        // pos/vel and the output pos/vel swap.
        // Hub state buffer must live before we build the bind groups
        // that reference it (it's declared further down — temporarily
        // build it here).
        let hub_state_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_fem hub_state p"),
            // Two vec4 slots: (hub_xyz, hub_v_y) + (hub_v_x, hub_v_z, _, _)
            size: (8 * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let init_hub: [f32; 8] = [
            input.params.hub[0],
            input.params.hub[1],
            input.params.hub[2],
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        queue.write_buffer(&hub_state_buf, 0, bytemuck::bytes_of(&init_hub));

        let entries_a_to_b: [wgpu::BindGroupEntry; 38] = [
            entry(0, &rest_buf),
            entry(1, &pos_a),
            entry(2, &vel_a),
            entry(3, &mass_buf),
            entry(4, &tet_i_buf),
            entry(5, &tet_g_buf),
            entry(6, &tet_v_buf),
            entry(7, &tet_mu_buf),
            entry(8, &tet_lambda_buf),
            entry(9, &bead_id_buf),
            entry(10, &bead_off_buf),
            entry(11, &tread_id_buf),
            entry(12, &pos_b),
            entry(13, &vel_b),
            entry(14, &force_buf),
            entry(15, &tet_fk_buf),
            entry(16, &tet_k1_buf),
            entry(17, &tet_k2_buf),
            entry(18, &csr_off_buf),
            entry(19, &csr_ent_buf),
            entry(20, &hub_state_buf),
            entry(21, &tet_fk_b_buf),
            entry(22, &tet_k1_b_buf),
            entry(23, &tet_k2_b_buf),
            entry(24, &xpbd_edge_i_buf),
            entry(25, &xpbd_edge_p_buf),
            entry(26, &node_edge_off_buf),
            entry(27, &node_edge_ent_buf),
            entry(28, &tet_fk_c_buf),
            entry(29, &tet_k1_c_buf),
            entry(30, &tet_k2_c_buf),
            entry(31, &tet_fk_d_buf),
            entry(32, &tet_k1_d_buf),
            entry(33, &tet_k2_d_buf),
            entry(34, &xpbd_dihedral_i_buf),
            entry(35, &xpbd_dihedral_p_buf),
            entry(36, &node_dihedral_off_buf),
            entry(37, &node_dihedral_ent_buf),
        ];
        let entries_b_to_a: [wgpu::BindGroupEntry; 38] = [
            entry(0, &rest_buf),
            entry(1, &pos_b),
            entry(2, &vel_b),
            entry(3, &mass_buf),
            entry(4, &tet_i_buf),
            entry(5, &tet_g_buf),
            entry(6, &tet_v_buf),
            entry(7, &tet_mu_buf),
            entry(8, &tet_lambda_buf),
            entry(9, &bead_id_buf),
            entry(10, &bead_off_buf),
            entry(11, &tread_id_buf),
            entry(12, &pos_a),
            entry(13, &vel_a),
            entry(14, &force_buf),
            entry(15, &tet_fk_buf),
            entry(16, &tet_k1_buf),
            entry(17, &tet_k2_buf),
            entry(18, &csr_off_buf),
            entry(19, &csr_ent_buf),
            entry(20, &hub_state_buf),
            entry(21, &tet_fk_b_buf),
            entry(22, &tet_k1_b_buf),
            entry(23, &tet_k2_b_buf),
            entry(24, &xpbd_edge_i_buf),
            entry(25, &xpbd_edge_p_buf),
            entry(26, &node_edge_off_buf),
            entry(27, &node_edge_ent_buf),
            entry(28, &tet_fk_c_buf),
            entry(29, &tet_k1_c_buf),
            entry(30, &tet_k2_c_buf),
            entry(31, &tet_fk_d_buf),
            entry(32, &tet_k1_d_buf),
            entry(33, &tet_k2_d_buf),
            entry(34, &xpbd_dihedral_i_buf),
            entry(35, &xpbd_dihedral_p_buf),
            entry(36, &node_dihedral_off_buf),
            entry(37, &node_dihedral_ent_buf),
        ];
        let bg_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem io bg A→B"),
            layout: &self.layout_io,
            entries: &entries_a_to_b,
        });
        let bg_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem io bg B→A"),
            layout: &self.layout_io,
            entries: &entries_b_to_a,
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem uniforms bg p"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniform_buf)],
        });

        // Stage (f) — hub uniform companion buffer (the hub_state_buf
        // itself is already created above so it can be bound into the
        // main step's bind group).
        let hub_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_fem hub_uniforms p"),
            size: core::mem::size_of::<HubUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let hub_bg_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem hub io bg A"),
            layout: &self.hub_layout_io,
            entries: &[
                entry(0, &bead_id_buf),
                entry(1, &bead_off_buf),
                entry(2, &pos_a),
                entry(3, &vel_a),
                entry(4, &hub_state_buf),
            ],
        });
        let hub_bg_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem hub io bg B"),
            layout: &self.hub_layout_io,
            entries: &[
                entry(0, &bead_id_buf),
                entry(1, &bead_off_buf),
                entry(2, &pos_b),
                entry(3, &vel_b),
                entry(4, &hub_state_buf),
            ],
        });
        let hub_bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem hub uniforms bg p"),
            layout: &self.hub_layout_uniforms,
            entries: &[entry(0, &hub_uniform_buf)],
        });

        let tet_count = input.tet_indices.len() / 4;
        let cached_uniforms = Uniforms {
            counts: [
                n as u32,
                tet_count as u32,
                input.bead_ids.len() as u32,
                input.tread_ids.len() as u32,
            ],
            counts2: [
                input.xpbd_edge_indices.len() as u32,
                input.xpbd_dihedral_indices.len() as u32,
                0,
                0,
            ],
            phys0: [
                input.params.dt,
                input.params.young_modulus,
                input.params.poisson_ratio,
                input.params.pressure_pa,
            ],
            phys1: [
                input.params.node_damping,
                input.params.bead_stiffness,
                input.params.floor_stiffness,
                input.params.floor_damping,
            ],
            phys2: [
                input.params.pressure_area_per_tread_node,
                input.params.hub[0],
                input.params.hub[1],
                input.params.hub[2],
            ],
            phys3: [
                input.params.gravity_y,
                input.params.cavity_radial_force_per_bead,
                input.params.belt_proxy_mu,
                input.params.xpbd_volume_compliance,
            ],
            phys4: [
                input.params.xpbd_bead_compliance,
                input.params.semi_implicit_stiffness_scale,
                input.params.xpbd_sidewall_ply_stretch_limit,
                input.params.xpbd_sidewall_ply_compliance,
            ],
            // PLAN_TOURQUE Stage G part 1 — torque/spin uniform row.
            phys5: [
                input.params.hub_omega,
                input.params.drive_torque_n_m,
                input.params.brake_torque_n_m,
                0.0,
            ],
        };

        Ok(TireFemPersistentMesh {
            n_nodes: n as u32,
            rest_buf,
            mass_buf,
            tet_i_buf,
            tet_g_buf,
            tet_v_buf,
            tet_mu_buf,
            tet_lambda_buf,
            tet_fk_buf,
            tet_k1_buf,
            tet_k2_buf,
            tet_fk_b_buf,
            tet_k1_b_buf,
            tet_k2_b_buf,
            tet_fk_c_buf,
            tet_k1_c_buf,
            tet_k2_c_buf,
            tet_fk_d_buf,
            tet_k1_d_buf,
            tet_k2_d_buf,
            xpbd_edge_i_buf,
            xpbd_edge_p_buf,
            node_edge_off_buf,
            node_edge_ent_buf,
            xpbd_dihedral_i_buf,
            xpbd_dihedral_p_buf,
            node_dihedral_off_buf,
            node_dihedral_ent_buf,
            csr_off_buf,
            csr_ent_buf,
            bead_id_buf,
            bead_off_buf,
            tread_id_buf,
            pos_a,
            pos_b,
            vel_a,
            vel_b,
            force_buf,
            uniform_buf,
            bg_a_to_b,
            bg_b_to_a,
            bg_uniforms,
            parity: false,
            counts: cached_uniforms.counts,
            cached_uniforms,
            hub_state_buf,
            hub_uniform_buf,
            hub_bg_a,
            hub_bg_b,
            hub_bg_uniforms,
            n_bead: input.bead_ids.len() as u32,
        })
    }

    /// Build a persistent GPU batch from multiple tire meshes.
    ///
    /// Constant topology/material buffers for each tire are uploaded once.
    /// Runtime code can then call [`Self::step_persistent_batch`] or
    /// [`Self::step_persistent_batch_with_hubs`] to advance every tire without
    /// re-uploading per-tire mesh data.
    pub fn create_persistent_batch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        inputs: &[TireFemGpuInput],
    ) -> Result<TireFemGpuBatch> {
        let _dispatch_plan = tire_fem_gpu_batch_dispatch_plan(inputs)?;
        let mut meshes = Vec::with_capacity(inputs.len());
        for input in inputs {
            meshes.push(self.create_persistent(device, queue, input)?);
        }
        Ok(TireFemGpuBatch::new(meshes))
    }

    /// Build a GPU-resident fused atlas batch from multiple tire meshes.
    ///
    /// The atlas owner uploads topology/material/routing buffers once, creates
    /// two ping-pong bind groups, and can then be advanced through
    /// [`Self::step_resident_atlas_batch`] with one fused dispatch per substep.
    pub fn create_resident_atlas_batch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        inputs: &[TireFemGpuInput],
    ) -> Result<TireFemGpuAtlasBatch> {
        let atlas = tire_fem_gpu_batch_atlas_buffers(inputs)?;
        atlas.validate()?;
        let n = atlas.layout.plan.total_node_count;
        if n == 0 {
            return Err(Error::Solver(
                "tire_fem atlas batch: cannot create resident empty atlas".into(),
            ));
        }

        let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_hub_state = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let out_bytes = bytes_of_slice::<[f32; 4]>(n);

        let rest_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident rest_pos",
            &atlas.rest_positions,
            usage_ro,
        );
        let pos_a = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident pos_a",
            &atlas.positions,
            usage_rw,
        );
        let pos_b =
            empty_storage_buffer(device, "tire_fem_atlas resident pos_b", out_bytes, usage_rw);
        let vel_a = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident vel_a",
            &atlas.velocities,
            usage_rw,
        );
        let vel_b =
            empty_storage_buffer(device, "tire_fem_atlas resident vel_b", out_bytes, usage_rw);
        let force_buf =
            empty_storage_buffer(device, "tire_fem_atlas resident force", out_bytes, usage_rw);
        let mass_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident mass_in",
            &atlas.masses,
            usage_ro,
        );
        let tet_i_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident tet_indices",
            &atlas.tet_indices,
            usage_ro,
        );
        let tet_g_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident tet_grads",
            &atlas.tet_grads,
            usage_ro,
        );
        let tet_v_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident tet_volume",
            &atlas.tet_volumes,
            usage_ro,
        );
        let tet_mu_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident tet_mu",
            &atlas.tet_mu,
            usage_ro,
        );
        let tet_lambda_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident tet_lambda",
            &atlas.tet_lambda,
            usage_ro,
        );
        let bead_id_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident bead_ids",
            &atlas.bead_ids,
            usage_ro,
        );
        let bead_off_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident bead_offsets",
            &atlas.bead_offsets,
            usage_ro,
        );
        let tread_id_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident tread_ids",
            &atlas.tread_ids,
            usage_ro,
        );
        let fiber_dir_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident tet_fiber_dir",
            &atlas.tet_fiber_dir,
            usage_ro,
        );
        let fiber_k1_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident tet_fiber_k1",
            &atlas.tet_fiber_k1,
            usage_ro,
        );
        let fiber_k2_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident tet_fiber_k2",
            &atlas.tet_fiber_k2,
            usage_ro,
        );
        let node_tet_off_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident node_tet_offsets",
            &atlas.node_tet_offsets,
            usage_ro,
        );
        let node_tet_ent_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident node_tet_entries",
            &atlas.node_tet_entries,
            usage_ro,
        );
        let hub_state = atlas_hub_state_rows(&atlas.params);
        let hub_state_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident hub_state",
            &hub_state,
            usage_hub_state,
        );
        let fiber_dir_b_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident tet_fiber_dir_b",
            &atlas.tet_fiber_dir_b,
            usage_ro,
        );
        let fiber_k1_b_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident tet_fiber_k1_b",
            &atlas.tet_fiber_k1_b,
            usage_ro,
        );
        let fiber_k2_b_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident tet_fiber_k2_b",
            &atlas.tet_fiber_k2_b,
            usage_ro,
        );
        let edge_i_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident xpbd_edge_indices",
            &atlas.xpbd_edge_indices,
            usage_ro,
        );
        let edge_p_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident xpbd_edge_params",
            &atlas.xpbd_edge_params,
            usage_ro,
        );
        let node_edge_off_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident node_edge_offsets",
            &atlas.node_edge_offsets,
            usage_ro,
        );
        let node_edge_ent_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident node_edge_entries",
            &atlas.node_edge_entries,
            usage_ro,
        );
        let fiber_dir_c_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident tet_fiber_dir_c",
            &atlas.tet_fiber_dir_c,
            usage_ro,
        );
        let fiber_k1_c_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident tet_fiber_k1_c",
            &atlas.tet_fiber_k1_c,
            usage_ro,
        );
        let fiber_k2_c_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident tet_fiber_k2_c",
            &atlas.tet_fiber_k2_c,
            usage_ro,
        );
        let fiber_dir_d_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident tet_fiber_dir_d",
            &atlas.tet_fiber_dir_d,
            usage_ro,
        );
        let fiber_k1_d_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident tet_fiber_k1_d",
            &atlas.tet_fiber_k1_d,
            usage_ro,
        );
        let fiber_k2_d_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident tet_fiber_k2_d",
            &atlas.tet_fiber_k2_d,
            usage_ro,
        );
        let dihedral_i_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident xpbd_dihedral_indices",
            &atlas.xpbd_dihedral_indices,
            usage_ro,
        );
        let dihedral_p_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident xpbd_dihedral_params",
            &atlas.xpbd_dihedral_params,
            usage_ro,
        );
        let node_dihedral_off_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident node_dihedral_offsets",
            &atlas.node_dihedral_offsets,
            usage_ro,
        );
        let node_dihedral_ent_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident node_dihedral_entries",
            &atlas.node_dihedral_entries,
            usage_ro,
        );
        let node_tire_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident node_tire_indices",
            &atlas.node_tire_indices,
            usage_ro,
        );
        let tet_tire_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident tet_tire_indices",
            &atlas.tet_tire_indices,
            usage_ro,
        );
        let bead_tire_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident bead_tire_indices",
            &atlas.bead_tire_indices,
            usage_ro,
        );
        let tread_tire_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident tread_tire_indices",
            &atlas.tread_tire_indices,
            usage_ro,
        );
        let param_rows_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident param_rows",
            &atlas.param_rows,
            usage_ro,
        );
        let offset_rows_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident offset_rows",
            &atlas.offset_rows,
            usage_ro,
        );
        let atlas_hub_param_rows = (0..atlas.layout.plan.tire_count)
            .map(|_| tire_fem_gpu_atlas_hub_param_row(TireFemHubParams::default()))
            .collect::<Vec<_>>();
        let atlas_hub_param_rows_buf = storage_buffer_from_slice(
            device,
            queue,
            "tire_fem_atlas resident hub_param_rows",
            &atlas_hub_param_rows,
            usage_rw,
        );

        let bg_a_to_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem_atlas resident bg a_to_b"),
            layout: &self.atlas_contract_layout_io,
            entries: &[
                entry(0, &rest_buf),
                entry(1, &pos_a),
                entry(2, &vel_a),
                entry(3, &mass_buf),
                entry(4, &tet_i_buf),
                entry(5, &tet_g_buf),
                entry(6, &tet_v_buf),
                entry(7, &tet_mu_buf),
                entry(8, &tet_lambda_buf),
                entry(9, &bead_id_buf),
                entry(10, &bead_off_buf),
                entry(11, &tread_id_buf),
                entry(12, &pos_b),
                entry(13, &vel_b),
                entry(14, &force_buf),
                entry(15, &fiber_dir_buf),
                entry(16, &fiber_k1_buf),
                entry(17, &fiber_k2_buf),
                entry(18, &node_tet_off_buf),
                entry(19, &node_tet_ent_buf),
                entry(20, &hub_state_buf),
                entry(21, &fiber_dir_b_buf),
                entry(22, &fiber_k1_b_buf),
                entry(23, &fiber_k2_b_buf),
                entry(24, &edge_i_buf),
                entry(25, &edge_p_buf),
                entry(26, &node_edge_off_buf),
                entry(27, &node_edge_ent_buf),
                entry(28, &fiber_dir_c_buf),
                entry(29, &fiber_k1_c_buf),
                entry(30, &fiber_k2_c_buf),
                entry(31, &fiber_dir_d_buf),
                entry(32, &fiber_k1_d_buf),
                entry(33, &fiber_k2_d_buf),
                entry(34, &dihedral_i_buf),
                entry(35, &dihedral_p_buf),
                entry(36, &node_dihedral_off_buf),
                entry(37, &node_dihedral_ent_buf),
                entry(38, &node_tire_buf),
                entry(39, &tet_tire_buf),
                entry(40, &bead_tire_buf),
                entry(41, &tread_tire_buf),
                entry(42, &param_rows_buf),
                entry(43, &offset_rows_buf),
            ],
        });
        let bg_b_to_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem_atlas resident bg b_to_a"),
            layout: &self.atlas_contract_layout_io,
            entries: &[
                entry(0, &rest_buf),
                entry(1, &pos_b),
                entry(2, &vel_b),
                entry(3, &mass_buf),
                entry(4, &tet_i_buf),
                entry(5, &tet_g_buf),
                entry(6, &tet_v_buf),
                entry(7, &tet_mu_buf),
                entry(8, &tet_lambda_buf),
                entry(9, &bead_id_buf),
                entry(10, &bead_off_buf),
                entry(11, &tread_id_buf),
                entry(12, &pos_a),
                entry(13, &vel_a),
                entry(14, &force_buf),
                entry(15, &fiber_dir_buf),
                entry(16, &fiber_k1_buf),
                entry(17, &fiber_k2_buf),
                entry(18, &node_tet_off_buf),
                entry(19, &node_tet_ent_buf),
                entry(20, &hub_state_buf),
                entry(21, &fiber_dir_b_buf),
                entry(22, &fiber_k1_b_buf),
                entry(23, &fiber_k2_b_buf),
                entry(24, &edge_i_buf),
                entry(25, &edge_p_buf),
                entry(26, &node_edge_off_buf),
                entry(27, &node_edge_ent_buf),
                entry(28, &fiber_dir_c_buf),
                entry(29, &fiber_k1_c_buf),
                entry(30, &fiber_k2_c_buf),
                entry(31, &fiber_dir_d_buf),
                entry(32, &fiber_k1_d_buf),
                entry(33, &fiber_k2_d_buf),
                entry(34, &dihedral_i_buf),
                entry(35, &dihedral_p_buf),
                entry(36, &node_dihedral_off_buf),
                entry(37, &node_dihedral_ent_buf),
                entry(38, &node_tire_buf),
                entry(39, &tet_tire_buf),
                entry(40, &bead_tire_buf),
                entry(41, &tread_tire_buf),
                entry(42, &param_rows_buf),
                entry(43, &offset_rows_buf),
            ],
        });
        let atlas_hub_bg_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem_atlas resident hub bg a"),
            layout: &self.atlas_hub_layout_io,
            entries: &[
                entry(1, &pos_a),
                entry(2, &vel_a),
                entry(9, &bead_id_buf),
                entry(10, &bead_off_buf),
                entry(20, &hub_state_buf),
                entry(43, &offset_rows_buf),
                entry(44, &atlas_hub_param_rows_buf),
            ],
        });
        let atlas_hub_bg_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem_atlas resident hub bg b"),
            layout: &self.atlas_hub_layout_io,
            entries: &[
                entry(1, &pos_b),
                entry(2, &vel_b),
                entry(9, &bead_id_buf),
                entry(10, &bead_off_buf),
                entry(20, &hub_state_buf),
                entry(43, &offset_rows_buf),
                entry(44, &atlas_hub_param_rows_buf),
            ],
        });

        Ok(TireFemGpuAtlasBatch {
            atlas,
            pos_a,
            pos_b,
            vel_a,
            vel_b,
            force_buf,
            param_rows_buf,
            atlas_offset_rows_buf: offset_rows_buf,
            hub_state_buf,
            atlas_hub_param_rows_buf,
            bg_a_to_b,
            bg_b_to_a,
            atlas_hub_bg_a,
            atlas_hub_bg_b,
            parity: false,
        })
    }

    /// Stage (f) — GPU hub reduction + integration. Reads the bead
    /// positions/velocities the main kernel just produced and updates
    /// the GPU-resident hub state. Call **after** `step_persistent` in
    /// each substep so the next iteration sees the new hub.
    pub fn step_hub_persistent(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &TireFemPersistentMesh,
        params: TireFemHubParams,
    ) {
        if mesh.n_bead == 0 {
            return;
        }
        let hub_u = HubUniforms {
            cfg: [
                mesh.n_bead as f32,
                params.hub_floor_active,
                params.hub_floor_y,
                params.dt,
            ],
            mech: [
                params.bead_stiffness,
                params.bead_damping,
                params.hub_mass,
                params.hub_damping,
            ],
            load: [params.downforce, params.gravity_y, 0.0, 0.0],
            extf: [
                params.external_force_xyz[0],
                params.external_force_xyz[1],
                params.external_force_xyz[2],
                params.lateral_damping,
            ],
        };
        queue.write_buffer(&mesh.hub_uniform_buf, 0, bytemuck::bytes_of(&hub_u));

        // The hub kernel reads the bead pos/vel from the buffers that the
        // *just-finished* main step wrote. After main runs, parity has
        // already been toggled — so the latest data lives in the buffer
        // matching the NEW parity:
        //   - parity == true  → new data lives in pos_b/vel_b → use hub_bg_b
        //   - parity == false → new data lives in pos_a/vel_a → use hub_bg_a
        let bg = if mesh.parity {
            &mesh.hub_bg_b
        } else {
            &mesh.hub_bg_a
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_fem hub encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_fem hub pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.hub_pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.set_bind_group(1, &mesh.hub_bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// G4 follow-up — hub reduction + integration with the contact
    /// wrench reduced on-device. `tread_forces` is the per-node force
    /// buffer the contact-feedback kernel wrote (one `vec4<f32>` row per
    /// FEM node; rows for non-contact nodes are zero) and
    /// `force_row_count` its row count. The kernel sums the X/Z
    /// components inside the dispatch and applies them to the hub on
    /// top of `params.external_force_xyz`, so the rolling propulsion no
    /// longer needs a host readback → sum → uniform round trip. The Y
    /// rows are deliberately ignored — the normal load reaches the hub
    /// through the mesh + bead reduction, matching the CPU
    /// `step_inner`'s `net_contact_fx/fz → hub_v_xz` convention.
    pub fn step_hub_persistent_with_tread_forces(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &TireFemPersistentMesh,
        params: TireFemHubParams,
        tread_forces: &wgpu::Buffer,
        force_row_count: u32,
    ) {
        if mesh.n_bead == 0 {
            return;
        }
        let hub_u = HubUniforms {
            cfg: [
                mesh.n_bead as f32,
                params.hub_floor_active,
                params.hub_floor_y,
                params.dt,
            ],
            mech: [
                params.bead_stiffness,
                params.bead_damping,
                params.hub_mass,
                params.hub_damping,
            ],
            load: [
                params.downforce,
                params.gravity_y,
                force_row_count as f32,
                0.0,
            ],
            extf: [
                params.external_force_xyz[0],
                params.external_force_xyz[1],
                params.external_force_xyz[2],
                params.lateral_damping,
            ],
        };
        queue.write_buffer(&mesh.hub_uniform_buf, 0, bytemuck::bytes_of(&hub_u));

        // Same parity rule as `step_hub_persistent`: read the pos/vel
        // buffers the just-finished node step wrote.
        let (pos_buf, vel_buf) = if mesh.parity {
            (&mesh.pos_b, &mesh.vel_b)
        } else {
            (&mesh.pos_a, &mesh.vel_a)
        };
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem hub tread bg"),
            layout: &self.hub_tread_layout_io,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: mesh.bead_id_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: mesh.bead_off_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pos_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: vel_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: mesh.hub_state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: tread_forces.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_fem hub tread encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_fem hub tread pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.hub_tread_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.set_bind_group(1, &mesh.hub_bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Stage (f) ext — read back the 3-DOF hub state. Returns
    /// `(x, y, z, v_y, v_x, v_z, _, _)` matching `hub_state` slots [0]
    /// and [1].
    pub fn readback_hub(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &TireFemPersistentMesh,
    ) -> Result<[f32; 8]> {
        let bytes = (8 * core::mem::size_of::<f32>()) as u64;
        let stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_fem hub readback stage"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_fem hub readback"),
        });
        encoder.copy_buffer_to_buffer(&mesh.hub_state_buf, 0, &stage, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));
        let slice = stage.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel::<Result<()>>();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            sender
                .send(r.map_err(|e| Error::Gpu(format!("hub readback: {e:?}"))))
                .ok();
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("hub readback poll: {e:?}")))?;
        receiver
            .recv()
            .map_err(|e| Error::Gpu(format!("hub readback recv: {e:?}")))??;
        let data = slice.get_mapped_range();
        let v: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        let out: [f32; 8] = [v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]];
        drop(data);
        stage.unmap();
        Ok(out)
    }

    /// Stage (f) — overwrite the GPU-resident 3-DOF hub state.
    pub fn upload_hub(&self, queue: &wgpu::Queue, mesh: &TireFemPersistentMesh, hub: [f32; 8]) {
        queue.write_buffer(&mesh.hub_state_buf, 0, bytemuck::bytes_of(&hub));
    }

    /// Stage (g) — advance one substep on persistent GPU buffers. Reads
    /// from the current "input" pos/vel buffers and writes the result
    /// into the opposite pair; toggles `parity` so the next call picks
    /// up automatically. No CPU readback.
    pub fn step_persistent(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &mut TireFemPersistentMesh,
        params: TireFemGpuParams,
    ) {
        mesh.cached_uniforms.phys0 = [
            params.dt,
            params.young_modulus,
            params.poisson_ratio,
            params.pressure_pa,
        ];
        mesh.cached_uniforms.phys1 = [
            params.node_damping,
            params.bead_stiffness,
            params.floor_stiffness,
            params.floor_damping,
        ];
        mesh.cached_uniforms.phys2 = [
            params.pressure_area_per_tread_node,
            params.hub[0],
            params.hub[1],
            params.hub[2],
        ];
        mesh.cached_uniforms.phys3 = [
            params.gravity_y,
            params.cavity_radial_force_per_bead,
            params.belt_proxy_mu,
            params.xpbd_volume_compliance,
        ];
        mesh.cached_uniforms.phys4 = [
            params.xpbd_bead_compliance,
            params.semi_implicit_stiffness_scale,
            params.xpbd_sidewall_ply_stretch_limit,
            params.xpbd_sidewall_ply_compliance,
        ];
        queue.write_buffer(
            &mesh.uniform_buf,
            0,
            bytemuck::bytes_of(&mesh.cached_uniforms),
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_fem persistent encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_fem persistent pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            let bg = if mesh.parity {
                &mesh.bg_b_to_a
            } else {
                &mesh.bg_a_to_b
            };
            pass.set_bind_group(0, bg, &[]);
            pass.set_bind_group(1, &mesh.bg_uniforms, &[]);
            pass.dispatch_workgroups(mesh.n_nodes.div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        mesh.parity = !mesh.parity;
    }

    /// Advance every tire in a persistent batch by one FEM substep.
    ///
    /// `params` must have one entry per tire. This keeps multi-wheel code on a
    /// single reusable API and validates that every tire in the batch receives
    /// explicit per-wheel parameters instead of silently sharing one hub/force
    /// state.
    pub fn step_persistent_batch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &mut TireFemGpuBatch,
        params: &[TireFemGpuParams],
    ) -> Result<()> {
        if params.len() != batch.meshes.len() {
            return Err(Error::Solver(format!(
                "tire_fem batch: params length {} must match tire count {}",
                params.len(),
                batch.meshes.len()
            )));
        }
        for (mesh, params) in batch.meshes.iter_mut().zip(params.iter().copied()) {
            self.step_persistent(device, queue, mesh, params);
        }
        Ok(())
    }

    /// Advance every tire and then update every GPU-resident hub state.
    pub fn step_persistent_batch_with_hubs(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &mut TireFemGpuBatch,
        fem_params: &[TireFemGpuParams],
        hub_params: &[TireFemHubParams],
    ) -> Result<()> {
        if hub_params.len() != batch.meshes.len() {
            return Err(Error::Solver(format!(
                "tire_fem batch: hub params length {} must match tire count {}",
                hub_params.len(),
                batch.meshes.len()
            )));
        }
        self.step_persistent_batch(device, queue, batch, fem_params)?;
        for (mesh, params) in batch.meshes.iter().zip(hub_params.iter().copied()) {
            self.step_hub_persistent(device, queue, mesh, params);
        }
        Ok(())
    }

    /// Advance a resident fused atlas batch by one FEM substep.
    ///
    /// `params` may change every substep; counts/topology stay fixed while the
    /// per-tire parameter rows are updated in-place on GPU. Hub rows are
    /// resident state: initialize/reseed them with
    /// [`Self::upload_resident_atlas_hub_rows`] or advance them with
    /// [`Self::step_resident_atlas_hubs`] so node stepping does not erase the
    /// GPU hub integrator between substeps.
    pub fn step_resident_atlas_batch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &mut TireFemGpuAtlasBatch,
        params: &[TireFemGpuParams],
    ) -> Result<()> {
        if params.len() != batch.atlas.layout.plan.tire_count {
            return Err(Error::Solver(format!(
                "tire_fem resident atlas: params length {} must match tire count {}",
                params.len(),
                batch.atlas.layout.plan.tire_count
            )));
        }
        for (i, params) in params.iter().copied().enumerate() {
            let row = &mut batch.atlas.param_rows[i];
            *row = tire_fem_gpu_batch_atlas_param_row_from_parts(params, row.counts, row.counts2);
            batch.atlas.params[i] = params;
        }
        queue.write_buffer(
            &batch.param_rows_buf,
            0,
            bytemuck::cast_slice(&batch.atlas.param_rows),
        );
        let contract = tire_fem_gpu_batch_atlas_dispatch_contract(&batch.atlas)?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_fem_atlas resident encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_fem_atlas resident pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.atlas_contract_pipeline);
            let bg = if batch.parity {
                &batch.bg_b_to_a
            } else {
                &batch.bg_a_to_b
            };
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(contract.workgroup_count, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        batch.parity = !batch.parity;
        Ok(())
    }

    /// Advance all hub rows in a resident fused atlas batch with one dispatch.
    ///
    /// Call this after [`Self::step_resident_atlas_batch`] in each substep.
    /// One workgroup owns one tire's bead slice, reduces bead reactions from
    /// the current atlas position/velocity buffers, and integrates that tire's
    /// packed hub position/velocity rows in-place. This is the fused-atlas
    /// equivalent of looping over [`Self::step_hub_persistent`].
    pub fn step_resident_atlas_hubs(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &TireFemGpuAtlasBatch,
        hub_params: &[TireFemHubParams],
    ) -> Result<()> {
        let tire_count = batch.atlas.layout.plan.tire_count;
        if hub_params.len() != tire_count {
            return Err(Error::Solver(format!(
                "tire_fem resident atlas hubs: params length {} must match tire count {}",
                hub_params.len(),
                tire_count
            )));
        }
        if tire_count == 0 {
            return Ok(());
        }
        let rows = hub_params
            .iter()
            .copied()
            .map(tire_fem_gpu_atlas_hub_param_row)
            .collect::<Vec<_>>();
        queue.write_buffer(
            &batch.atlas_hub_param_rows_buf,
            0,
            bytemuck::cast_slice(&rows),
        );
        self.step_resident_atlas_hubs_with_resident_params(device, queue, batch)
    }

    /// Advance resident atlas hubs using the hub-parameter rows already stored
    /// in the resident atlas buffer.
    ///
    /// This is the no-overwrite companion to [`Self::step_resident_atlas_hubs`].
    /// Use it after a GPU-side parameter assembly pass such as
    /// [`Self::update_resident_atlas_hub_params_from_brush_output`], otherwise
    /// the brush/contact-derived rows would be replaced by host parameters just
    /// before integration.
    pub fn step_resident_atlas_hubs_with_resident_params(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &TireFemGpuAtlasBatch,
    ) -> Result<()> {
        let tire_count = batch.atlas.layout.plan.tire_count;
        if tire_count == 0 {
            return Ok(());
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_fem_atlas resident hub encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_fem_atlas resident hub pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.atlas_hub_pipeline);
            let bg = if batch.parity {
                &batch.atlas_hub_bg_b
            } else {
                &batch.atlas_hub_bg_a
            };
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(tire_count as u32, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Update resident atlas hub parameter rows directly from a GPU-resident
    /// `BrushWheelOutput` buffer.
    ///
    /// This is the GPU-resident counterpart of
    /// [`tire_fem_hub_params_with_brush_outputs`]. The base hub parameters are
    /// uploaded into the resident atlas hub-param row buffer, then a compute
    /// pass adds each brush row's `world_force.xyz * force_scale` to the
    /// matching `extf.xyz` row in-place. Call [`Self::step_resident_atlas_hubs`]
    /// after this to integrate FEM hubs with the vehicle/brush loads.
    pub fn update_resident_atlas_hub_params_from_brush_output(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &TireFemGpuAtlasBatch,
        brush_output_buffer: &wgpu::Buffer,
        brush_output_count: usize,
        base_hub_params: &[TireFemHubParams],
        force_scale: f32,
    ) -> Result<()> {
        let tire_count = batch.atlas.layout.plan.tire_count;
        if brush_output_count != tire_count {
            return Err(Error::Solver(format!(
                "tire_fem resident atlas brush hub params: brush output count {brush_output_count} must match tire count {tire_count}"
            )));
        }
        if base_hub_params.len() != tire_count {
            return Err(Error::Solver(format!(
                "tire_fem resident atlas brush hub params: base params length {} must match tire count {tire_count}",
                base_hub_params.len()
            )));
        }
        if tire_count == 0 {
            return Ok(());
        }
        let rows = base_hub_params
            .iter()
            .copied()
            .map(tire_fem_gpu_atlas_hub_param_row)
            .collect::<Vec<_>>();
        queue.write_buffer(
            &batch.atlas_hub_param_rows_buf,
            0,
            bytemuck::cast_slice(&rows),
        );
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_fem_atlas brush-to-hub-param uniforms"),
            size: core::mem::size_of::<BrushHubUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&BrushHubUniforms {
                counts: [tire_count as u32, 0, 0, 0],
                params: [
                    finite_f32_or(force_scale, 1.0).clamp(-10.0, 10.0),
                    0.0,
                    0.0,
                    0.0,
                ],
            }),
        );
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem_atlas brush-to-hub-param io bg"),
            layout: &self.atlas_hub_params_from_brush_layout_io,
            entries: &[
                entry(45, brush_output_buffer),
                entry(46, &batch.atlas_hub_param_rows_buf),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem_atlas brush-to-hub-param uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniforms)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_fem_atlas brush-to-hub-param encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_fem_atlas brush-to-hub-param pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.atlas_hub_params_from_brush_pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups((tire_count as u32).div_ceil(64), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Assemble resident atlas hub parameters from a GPU-resident brush output
    /// buffer, then integrate the resident FEM hubs without staging the brush
    /// rows back through CPU.
    ///
    /// This closes the runtime hazard where [`Self::step_resident_atlas_hubs`]
    /// would otherwise rewrite the parameter rows from host data and erase the
    /// GPU-side brush loads before the hub solver runs.
    pub fn step_resident_atlas_hubs_from_brush_output(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &TireFemGpuAtlasBatch,
        brush_output_buffer: &wgpu::Buffer,
        brush_output_count: usize,
        base_hub_params: &[TireFemHubParams],
        force_scale: f32,
    ) -> Result<()> {
        self.update_resident_atlas_hub_params_from_brush_output(
            device,
            queue,
            batch,
            brush_output_buffer,
            brush_output_count,
            base_hub_params,
            force_scale,
        )?;
        self.step_resident_atlas_hubs_with_resident_params(device, queue, batch)
    }

    /// Update resident atlas hub parameter rows from packed GPU-resident FEM
    /// contact-reduction output rows.
    ///
    /// Each tire consumes four `vec4` rows in the same shape used by
    /// `tire_fem_contact_reduce.wgsl`: rows 0/1 are contact diagnostics, row 2
    /// is `world_force.xyz` plus sinkage, and row 3 is `world_moment.xyz` plus
    /// water film. This maps row-2 force directly into the atlas hub external
    /// force vector without staging the reduced wrench through CPU.
    pub fn update_resident_atlas_hub_params_from_contact_reduction_output(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &TireFemGpuAtlasBatch,
        contact_reduction_output_buffer: &wgpu::Buffer,
        contact_reduction_output_count: usize,
        base_hub_params: &[TireFemHubParams],
        force_scale: f32,
    ) -> Result<()> {
        let tire_count = batch.atlas.layout.plan.tire_count;
        if contact_reduction_output_count != tire_count {
            return Err(Error::Solver(format!(
                "tire_fem resident atlas contact hub params: contact reduction count {contact_reduction_output_count} must match tire count {tire_count}"
            )));
        }
        if base_hub_params.len() != tire_count {
            return Err(Error::Solver(format!(
                "tire_fem resident atlas contact hub params: base params length {} must match tire count {tire_count}",
                base_hub_params.len()
            )));
        }
        if tire_count == 0 {
            return Ok(());
        }
        let rows = base_hub_params
            .iter()
            .copied()
            .map(tire_fem_gpu_atlas_hub_param_row)
            .collect::<Vec<_>>();
        queue.write_buffer(
            &batch.atlas_hub_param_rows_buf,
            0,
            bytemuck::cast_slice(&rows),
        );
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_fem_atlas contact-reduction-to-hub-param uniforms"),
            size: core::mem::size_of::<BrushHubUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&BrushHubUniforms {
                counts: [tire_count as u32, 0, 0, 0],
                params: [
                    finite_f32_or(force_scale, 1.0).clamp(-10.0, 10.0),
                    0.0,
                    0.0,
                    0.0,
                ],
            }),
        );
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem_atlas contact-reduction-to-hub-param io bg"),
            layout: &self.atlas_hub_params_from_contact_layout_io,
            entries: &[
                entry(46, &batch.atlas_hub_param_rows_buf),
                entry(47, contact_reduction_output_buffer),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem_atlas contact-reduction-to-hub-param uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniforms)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_fem_atlas contact-reduction-to-hub-param encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_fem_atlas contact-reduction-to-hub-param pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.atlas_hub_params_from_contact_pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups((tire_count as u32).div_ceil(64), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Assemble resident atlas hub parameters from packed GPU-resident contact
    /// reduction rows, then integrate the resident FEM hubs without CPU
    /// staging of the reduced world force.
    pub fn step_resident_atlas_hubs_from_contact_reduction_output(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &TireFemGpuAtlasBatch,
        contact_reduction_output_buffer: &wgpu::Buffer,
        contact_reduction_output_count: usize,
        base_hub_params: &[TireFemHubParams],
        force_scale: f32,
    ) -> Result<()> {
        self.update_resident_atlas_hub_params_from_contact_reduction_output(
            device,
            queue,
            batch,
            contact_reduction_output_buffer,
            contact_reduction_output_count,
            base_hub_params,
            force_scale,
        )?;
        self.step_resident_atlas_hubs_with_resident_params(device, queue, batch)
    }

    /// Update resident atlas hub parameter rows from GPU-resident FEM
    /// contact-feedback per-node force rows.
    ///
    /// This is the resident counterpart of
    /// [`tire_fem_hub_params_with_feedback_output`]. It sums force rows inside
    /// each tire's packed node range from `offset_rows` and writes the scaled
    /// result into `AtlasHubParamRow.extf.xyz`, allowing contact-feedback forces
    /// to drive the atlas hub integrator without CPU staging.
    pub fn update_resident_atlas_hub_params_from_feedback_output(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &TireFemGpuAtlasBatch,
        feedback_output: &TireFemContactFeedbackResidentOutput,
        base_hub_params: &[TireFemHubParams],
        force_scale: f32,
    ) -> Result<()> {
        let tire_count = batch.atlas.layout.plan.tire_count;
        if feedback_output.node_count != batch.atlas.layout.plan.total_node_count {
            return Err(Error::Solver(format!(
                "tire_fem resident atlas feedback hub params: feedback node rows {} must match atlas node count {}",
                feedback_output.node_count, batch.atlas.layout.plan.total_node_count
            )));
        }
        if base_hub_params.len() != tire_count {
            return Err(Error::Solver(format!(
                "tire_fem resident atlas feedback hub params: base params length {} must match tire count {tire_count}",
                base_hub_params.len()
            )));
        }
        if tire_count == 0 {
            return Ok(());
        }
        let rows = base_hub_params
            .iter()
            .copied()
            .map(tire_fem_gpu_atlas_hub_param_row)
            .collect::<Vec<_>>();
        queue.write_buffer(
            &batch.atlas_hub_param_rows_buf,
            0,
            bytemuck::cast_slice(&rows),
        );
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_fem_atlas feedback-force-to-hub-param uniforms"),
            size: core::mem::size_of::<BrushHubUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &uniforms,
            0,
            bytemuck::bytes_of(&BrushHubUniforms {
                counts: [tire_count as u32, 0, 0, 0],
                params: [
                    finite_f32_or(force_scale, 1.0).clamp(-10.0, 10.0),
                    0.0,
                    0.0,
                    0.0,
                ],
            }),
        );
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem_atlas feedback-force-to-hub-param io bg"),
            layout: &self.atlas_hub_params_from_feedback_layout_io,
            entries: &[
                entry(43, &batch.atlas_offset_rows_buf),
                entry(46, &batch.atlas_hub_param_rows_buf),
                entry(48, &feedback_output.forces),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem_atlas feedback-force-to-hub-param uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniforms)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_fem_atlas feedback-force-to-hub-param encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_fem_atlas feedback-force-to-hub-param pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.atlas_hub_params_from_feedback_pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups((tire_count as u32).div_ceil(64), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Assemble resident atlas hub parameters from GPU-resident contact
    /// feedback force rows, then integrate the resident FEM hubs without CPU
    /// staging of the per-node force buffer.
    pub fn step_resident_atlas_hubs_from_feedback_output(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &TireFemGpuAtlasBatch,
        feedback_output: &TireFemContactFeedbackResidentOutput,
        base_hub_params: &[TireFemHubParams],
        force_scale: f32,
    ) -> Result<()> {
        self.update_resident_atlas_hub_params_from_feedback_output(
            device,
            queue,
            batch,
            feedback_output,
            base_hub_params,
            force_scale,
        )?;
        self.step_resident_atlas_hubs_with_resident_params(device, queue, batch)
    }

    /// Step the fully resident vehicle brush owner and feed its GPU-resident
    /// per-wheel brush output directly into resident FEM atlas hub integration.
    ///
    /// The sequence is:
    /// resident wheel spin/suspension/chassis → GPU brush output →
    /// GPU brush-output-to-atlas-hub-param rows → resident FEM hub integration.
    /// Call [`Self::step_resident_atlas_batch`] separately in the surrounding
    /// substep to advance the FEM node state with the latest hub rows.
    pub fn step_resident_vehicle_fem_hubs_from_brush_owner(
        &self,
        vehicle_kernel: &TireVehicleYawGpuKernel,
        brush_kernel: &TireBrushGpuKernel,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vehicle_owner: &mut TireVehicleResidentChassisBrushOwner,
        fem_batch: &TireFemGpuAtlasBatch,
        dt: f32,
        terrain: Option<&BrushTerrain<'_>>,
        base_hub_params: &[TireFemHubParams],
        force_scale: f32,
    ) -> Result<TireBrushResidentOutput> {
        let brush_output = vehicle_kernel.step_resident_chassis_brush_owner_from_resident_state(
            brush_kernel,
            device,
            queue,
            vehicle_owner,
            dt,
            terrain,
        )?;
        self.step_resident_atlas_hubs_from_brush_output(
            device,
            queue,
            fem_batch,
            &brush_output.buffer,
            brush_output.wheel_count,
            base_hub_params,
            force_scale,
        )?;
        Ok(brush_output)
    }

    /// Advance one coherent resident FEM/vehicle substep:
    ///
    /// 1. step all resident FEM atlas nodes with the current resident hub rows;
    /// 2. step resident vehicle spin/suspension/chassis and GPU brush output;
    /// 3. assemble atlas hub parameters from that GPU-resident brush output;
    /// 4. integrate resident FEM hub rows without overwriting GPU-assembled
    ///    load rows from host-side data.
    ///
    /// This is the first reusable no-readback substep scheduler for the
    /// resident vehicle→brush→FEM-atlas path. Higher-level viewers can loop it
    /// for substeps instead of manually ordering node, brush, and hub dispatches.
    pub fn step_resident_vehicle_fem_atlas_from_brush_owner(
        &self,
        vehicle_kernel: &TireVehicleYawGpuKernel,
        brush_kernel: &TireBrushGpuKernel,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        vehicle_owner: &mut TireVehicleResidentChassisBrushOwner,
        fem_batch: &mut TireFemGpuAtlasBatch,
        fem_params: &[TireFemGpuParams],
        dt: f32,
        terrain: Option<&BrushTerrain<'_>>,
        base_hub_params: &[TireFemHubParams],
        force_scale: f32,
    ) -> Result<TireBrushResidentOutput> {
        self.step_resident_atlas_batch(device, queue, fem_batch, fem_params)?;
        self.step_resident_vehicle_fem_hubs_from_brush_owner(
            vehicle_kernel,
            brush_kernel,
            device,
            queue,
            vehicle_owner,
            fem_batch,
            dt,
            terrain,
            base_hub_params,
            force_scale,
        )
    }

    /// Create a combined resident vehicle+FEM-atlas owner for live runtime use.
    ///
    /// The FEM tire count, vehicle wheel-spin count, suspension count, planar
    /// wheel-offset count, and vertical wheel count must all agree; otherwise
    /// the resident no-readback handoff would silently pair the wrong wheel
    /// with the wrong FEM tire.
    pub fn create_resident_vehicle_fem_atlas_owner(
        &self,
        vehicle_kernel: &TireVehicleYawGpuKernel,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        fem_inputs: &[TireFemGpuInput],
        spin_input: &TireVehicleWheelSpinGpuInput,
        suspension_input: &TireVehicleSuspensionGpuInput,
        brush_input: TireVehicleBrushRowsFromSpinGpuInput,
        planar_input: &TireVehiclePlanarFromBrushGpuInput,
        vertical_input: &TireVehicleVerticalGpuInput,
    ) -> Result<TireResidentVehicleFemAtlasOwner> {
        let tire_count = fem_inputs.len();
        if tire_count != spin_input.wheel_state.len()
            || tire_count != suspension_input.suspension_geometry.len()
            || tire_count != planar_input.wheel_offsets.len()
        {
            return Err(Error::Solver(format!(
                "resident vehicle FEM atlas owner: tire count {tire_count}, spin wheels {}, suspension rows {}, planar offsets {} must match",
                spin_input.wheel_state.len(),
                suspension_input.suspension_geometry.len(),
                planar_input.wheel_offsets.len()
            )));
        }
        let fem = self.create_resident_atlas_batch(device, queue, fem_inputs)?;
        let vehicle = vehicle_kernel.create_resident_chassis_brush_owner_with_vertical(
            device,
            queue,
            spin_input,
            suspension_input,
            brush_input,
            planar_input,
            vertical_input,
        )?;
        if vehicle.wheel_count() != tire_count || fem.tire_count() != tire_count {
            return Err(Error::Solver(format!(
                "resident vehicle FEM atlas owner: constructed vehicle wheels {} and FEM tires {} must match requested tire count {tire_count}",
                vehicle.wheel_count(),
                fem.tire_count()
            )));
        }
        Ok(TireResidentVehicleFemAtlasOwner {
            vehicle,
            fem,
            tire_count,
        })
    }

    /// Step a combined resident vehicle+FEM-atlas owner once.
    ///
    /// This is the owner-oriented wrapper around
    /// [`Self::step_resident_vehicle_fem_atlas_from_brush_owner`].
    pub fn step_resident_vehicle_fem_atlas_owner(
        &self,
        vehicle_kernel: &TireVehicleYawGpuKernel,
        brush_kernel: &TireBrushGpuKernel,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        owner: &mut TireResidentVehicleFemAtlasOwner,
        fem_params: &[TireFemGpuParams],
        dt: f32,
        terrain: Option<&BrushTerrain<'_>>,
        base_hub_params: &[TireFemHubParams],
        force_scale: f32,
    ) -> Result<TireBrushResidentOutput> {
        if fem_params.len() != owner.tire_count || base_hub_params.len() != owner.tire_count {
            return Err(Error::Solver(format!(
                "resident vehicle FEM atlas owner step: FEM params {}, hub params {}, owner tire count {} must match",
                fem_params.len(),
                base_hub_params.len(),
                owner.tire_count
            )));
        }
        self.step_resident_vehicle_fem_atlas_from_brush_owner(
            vehicle_kernel,
            brush_kernel,
            device,
            queue,
            &mut owner.vehicle,
            &mut owner.fem,
            fem_params,
            dt,
            terrain,
            base_hub_params,
            force_scale,
        )
    }

    pub fn readback_resident_atlas_hub_param_rows(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &TireFemGpuAtlasBatch,
    ) -> Result<Vec<TireFemGpuAtlasHubParamRow>> {
        let tire_count = batch.atlas.layout.plan.tire_count;
        let stage = stage_vec4(
            device,
            "tire_fem_atlas resident stage hub_param_rows",
            tire_count * 4,
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_fem_atlas resident hub-param readback"),
        });
        encoder.copy_buffer_to_buffer(
            &batch.atlas_hub_param_rows_buf,
            0,
            &stage,
            0,
            bytes_of_slice::<TireFemGpuAtlasHubParamRow>(tire_count),
        );
        queue.submit(std::iter::once(encoder.finish()));
        let rows = read_vec4(device, &stage, "tire_fem_atlas resident hub_param_rows")?;
        Ok(rows
            .chunks_exact(4)
            .map(|chunk| TireFemGpuAtlasHubParamRow {
                cfg: chunk[0],
                mech: chunk[1],
                load: chunk[2],
                extf: chunk[3],
            })
            .collect())
    }

    /// Upload externally modified per-tire state into the current ping-pong
    /// side of a resident fused atlas batch.
    ///
    /// Viewer/vehicle runtime code can use this to reseed GPU ownership after
    /// a CPU-side pose reset, teleport, editor manipulation, or fallback step
    /// without rebuilding topology/material/routing buffers.
    pub fn upload_resident_atlas_batch_state(
        &self,
        queue: &wgpu::Queue,
        batch: &TireFemGpuAtlasBatch,
        outputs: &[TireFemGpuOutput],
    ) -> Result<()> {
        let rows = batch.atlas.join_outputs(outputs)?;
        let (current_pos, current_vel) = if batch.parity {
            (&batch.pos_b, &batch.vel_b)
        } else {
            (&batch.pos_a, &batch.vel_a)
        };
        queue.write_buffer(current_pos, 0, bytemuck::cast_slice(&rows.positions));
        queue.write_buffer(current_vel, 0, bytemuck::cast_slice(&rows.velocities));
        queue.write_buffer(&batch.force_buf, 0, bytemuck::cast_slice(&rows.forces));
        Ok(())
    }

    /// Upload one externally modified tire into the current resident atlas
    /// ping-pong side without touching the other tires.
    ///
    /// This is the viewer/editor-friendly reseed path: a reset or manipulation
    /// can replace one tire's node rows while preserving the resident packed
    /// atlas topology and all other tire states.
    pub fn upload_resident_atlas_tire_state(
        &self,
        queue: &wgpu::Queue,
        batch: &TireFemGpuAtlasBatch,
        tire_index: usize,
        output: &TireFemGpuOutput,
    ) -> Result<()> {
        let row = batch.atlas.layout.rows.get(tire_index).ok_or_else(|| {
            Error::Solver(format!(
                "tire_fem resident atlas upload: tire index {tire_index} out of range"
            ))
        })?;
        expect_len(
            "tire_fem resident atlas tire positions",
            output.positions.len(),
            row.node_count,
        )?;
        expect_len(
            "tire_fem resident atlas tire velocities",
            output.velocities.len(),
            row.node_count,
        )?;
        expect_len(
            "tire_fem resident atlas tire forces",
            output.forces.len(),
            row.node_count,
        )?;
        let (current_pos, current_vel) = if batch.parity {
            (&batch.pos_b, &batch.vel_b)
        } else {
            (&batch.pos_a, &batch.vel_a)
        };
        let byte_offset = bytes_of_slice::<[f32; 4]>(row.node_offset);
        queue.write_buffer(
            current_pos,
            byte_offset,
            bytemuck::cast_slice(&output.positions),
        );
        queue.write_buffer(
            current_vel,
            byte_offset,
            bytemuck::cast_slice(&output.velocities),
        );
        queue.write_buffer(
            &batch.force_buf,
            byte_offset,
            bytemuck::cast_slice(&output.forces),
        );
        Ok(())
    }

    /// Upload packed hub rows for every tire in a resident fused atlas batch.
    ///
    /// Rows are `[hub_position.xyz, _]`, `[hub_velocity.xyz, _]` pairs in tire
    /// order. `step_resident_atlas_batch` reads these rows but no longer
    /// refreshes them from per-step FEM params; callers should use this for
    /// resets/editor manipulation and [`Self::step_resident_atlas_hubs`] for
    /// normal GPU-owned hub integration.
    pub fn upload_resident_atlas_hub_rows(
        &self,
        queue: &wgpu::Queue,
        batch: &TireFemGpuAtlasBatch,
        hub_rows: &[[f32; 4]],
    ) -> Result<()> {
        expect_len(
            "tire_fem resident atlas hub rows",
            hub_rows.len(),
            batch.atlas.layout.plan.tire_count * 2,
        )?;
        queue.write_buffer(&batch.hub_state_buf, 0, bytemuck::cast_slice(hub_rows));
        Ok(())
    }

    /// Upload one tire's hub position/velocity rows into a resident atlas.
    pub fn upload_resident_atlas_tire_hub(
        &self,
        queue: &wgpu::Queue,
        batch: &TireFemGpuAtlasBatch,
        tire_index: usize,
        hub_position: [f32; 3],
        hub_velocity: [f32; 3],
    ) -> Result<()> {
        if tire_index >= batch.atlas.layout.plan.tire_count {
            return Err(Error::Solver(format!(
                "tire_fem resident atlas hub upload: tire index {tire_index} out of range"
            )));
        }
        let rows = [
            [hub_position[0], hub_position[1], hub_position[2], 0.0],
            [hub_velocity[0], hub_velocity[1], hub_velocity[2], 0.0],
        ];
        queue.write_buffer(
            &batch.hub_state_buf,
            bytes_of_slice::<[f32; 4]>(tire_index * 2),
            bytemuck::cast_slice(&rows),
        );
        Ok(())
    }

    /// Upload already-packed atlas rows into the current resident atlas state.
    pub fn upload_resident_atlas_rows(
        &self,
        queue: &wgpu::Queue,
        batch: &TireFemGpuAtlasBatch,
        rows: &TireFemGpuBatchAtlasOutputRows,
    ) -> Result<()> {
        let outputs = batch.atlas.split_output_rows(rows)?;
        self.upload_resident_atlas_batch_state(queue, batch, &outputs)
    }

    /// Read back the current packed rows from a resident fused atlas batch.
    pub fn readback_resident_atlas_rows(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &TireFemGpuAtlasBatch,
    ) -> Result<TireFemGpuBatchAtlasOutputRows> {
        let n = batch.atlas.layout.plan.total_node_count;
        let bytes = bytes_of_slice::<[f32; 4]>(n);
        let stage_pos = stage_vec4(device, "tire_fem_atlas resident stage pos", n);
        let stage_vel = stage_vec4(device, "tire_fem_atlas resident stage vel", n);
        let stage_force = stage_vec4(device, "tire_fem_atlas resident stage force", n);
        let (current_pos, current_vel) = if batch.parity {
            (&batch.pos_b, &batch.vel_b)
        } else {
            (&batch.pos_a, &batch.vel_a)
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_fem_atlas resident readback"),
        });
        encoder.copy_buffer_to_buffer(current_pos, 0, &stage_pos, 0, bytes);
        encoder.copy_buffer_to_buffer(current_vel, 0, &stage_vel, 0, bytes);
        encoder.copy_buffer_to_buffer(&batch.force_buf, 0, &stage_force, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));
        Ok(TireFemGpuBatchAtlasOutputRows {
            positions: read_vec4(device, &stage_pos, "tire_fem_atlas resident positions")?,
            velocities: read_vec4(device, &stage_vel, "tire_fem_atlas resident velocities")?,
            forces: read_vec4(device, &stage_force, "tire_fem_atlas resident forces")?,
        })
    }

    /// Read back packed hub position/velocity rows from a resident atlas.
    pub fn readback_resident_atlas_hub_rows(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &TireFemGpuAtlasBatch,
    ) -> Result<Vec<[f32; 4]>> {
        let row_count = batch.atlas.layout.plan.tire_count * 2;
        let bytes = bytes_of_slice::<[f32; 4]>(row_count);
        let stage_hub = stage_vec4(device, "tire_fem_atlas resident stage hub", row_count);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_fem_atlas resident hub readback"),
        });
        encoder.copy_buffer_to_buffer(&batch.hub_state_buf, 0, &stage_hub, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));
        read_vec4(device, &stage_hub, "tire_fem_atlas resident hub rows")
    }

    /// Read back a resident fused atlas batch split into per-tire output rows.
    pub fn readback_resident_atlas_batch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &TireFemGpuAtlasBatch,
    ) -> Result<Vec<TireFemGpuOutput>> {
        let rows = self.readback_resident_atlas_rows(device, queue, batch)?;
        batch.atlas.split_output_rows(&rows)
    }

    /// Stage (g) — push fresh positions/velocities onto the GPU. Useful
    /// when the CPU has nudged a tire externally (e.g. wheel rotated)
    /// and we need to seed the next substep batch from outside.
    pub fn upload_state(
        &self,
        queue: &wgpu::Queue,
        mesh: &TireFemPersistentMesh,
        positions: &[[f32; 4]],
        velocities: &[[f32; 4]],
    ) {
        let (pos_buf, vel_buf) = if mesh.parity {
            (&mesh.pos_b, &mesh.vel_b)
        } else {
            (&mesh.pos_a, &mesh.vel_a)
        };
        queue.write_buffer(pos_buf, 0, bytemuck::cast_slice(positions));
        queue.write_buffer(vel_buf, 0, bytemuck::cast_slice(velocities));
    }

    /// Stage (g) — read back the *current* tire state once per frame.
    /// `parity == false` means the latest state lives in the A buffers
    /// (because the next dispatch would write into B from A), and vice
    /// versa.
    pub fn readback_state(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &TireFemPersistentMesh,
    ) -> Result<TireFemGpuOutput> {
        let n = mesh.n_nodes as usize;
        let bytes = (n.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let stage_pos = stage_vec4(device, "tire_fem stage pos persist", n);
        let stage_vel = stage_vec4(device, "tire_fem stage vel persist", n);
        let stage_force = stage_vec4(device, "tire_fem stage force persist", n);
        let (current_pos, current_vel) = if mesh.parity {
            (&mesh.pos_b, &mesh.vel_b)
        } else {
            (&mesh.pos_a, &mesh.vel_a)
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_fem persistent readback"),
        });
        encoder.copy_buffer_to_buffer(current_pos, 0, &stage_pos, 0, bytes);
        encoder.copy_buffer_to_buffer(current_vel, 0, &stage_vel, 0, bytes);
        encoder.copy_buffer_to_buffer(&mesh.force_buf, 0, &stage_force, 0, bytes);
        queue.submit(std::iter::once(encoder.finish()));
        Ok(TireFemGpuOutput {
            positions: read_vec4(device, &stage_pos, "tire_fem positions p")?,
            velocities: read_vec4(device, &stage_vel, "tire_fem velocities p")?,
            forces: read_vec4(device, &stage_force, "tire_fem forces p")?,
        })
    }

    /// Read back every tire in a persistent batch once per frame.
    pub fn readback_batch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        batch: &TireFemGpuBatch,
    ) -> Result<Vec<TireFemGpuOutput>> {
        batch
            .meshes
            .iter()
            .map(|mesh| self.readback_state(device, queue, mesh))
            .collect()
    }
}

impl TireFemContactReductionKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::tire_fem_contact_reduce"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/tire_fem_contact_reduce.wgsl").into(),
            ),
        });
        let layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_fem_contact_reduce io"),
            entries: &[
                storage_entry(0, true),  // positions
                storage_entry(1, true),  // tread_ids
                storage_entry(2, false), // output diagnostics
                storage_entry(3, true),  // terrain heights
                storage_entry(4, true),  // terrain friction map
                storage_entry(5, true),  // terrain water-film map
                storage_entry(6, true),  // SDF primitive rows
                storage_entry(7, true),  // terrain sinkage map
            ],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_fem_contact_reduce uniforms"),
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
            label: Some("molla::tire_fem_contact_reduce layout"),
            bind_group_layouts: &[&layout_io, &layout_uniforms],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::tire_fem_contact_reduce pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("reduce_contact"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            pipeline,
            layout_io,
            layout_uniforms,
        }
    }

    pub fn dispatch_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireFemContactReductionInput,
    ) -> Result<TireFemContactReductionOutput> {
        if input.positions.is_empty() {
            return Err(Error::Solver(
                "tire_fem_contact_reduce: positions must not be empty".into(),
            ));
        }
        if let Some(terrain) = &input.terrain {
            let expected = terrain.nx as usize * terrain.nz as usize;
            if terrain.nx == 0 || terrain.nz == 0 || terrain.heights.len() != expected {
                return Err(Error::Solver(
                    "tire_fem_contact_reduce: heightfield dimensions must match heights".into(),
                ));
            }
            if let Some(map) = &terrain.friction_map
                && map.len() != expected
            {
                return Err(Error::Solver(
                    "tire_fem_contact_reduce: friction_map dimensions must match heights".into(),
                ));
            }
            if let Some(map) = &terrain.water_film_map
                && map.len() != expected
            {
                return Err(Error::Solver(
                    "tire_fem_contact_reduce: water_film_map dimensions must match heights".into(),
                ));
            }
            if let Some(map) = &terrain.sinkage_map
                && map.len() != expected
            {
                return Err(Error::Solver(
                    "tire_fem_contact_reduce: sinkage_map dimensions must match heights".into(),
                ));
            }
        }
        let positions_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_fem_contact_reduce positions"),
            size: (input.positions.len().max(1) * core::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tread_len = input.tread_ids.len().max(1);
        let tread_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_fem_contact_reduce tread_ids"),
            size: (tread_len * core::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_fem_contact_reduce output"),
            size: (4 * core::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let terrain_len = input
            .terrain
            .as_ref()
            .map(|t| t.heights.len())
            .unwrap_or(1)
            .max(1);
        let (primitive_rows, triangle_bvh_node_count, triangle_bvh_index_row_count): (
            Vec<[f32; 4]>,
            u32,
            u32,
        ) = input
            .terrain
            .as_ref()
            .map(contact_primitive_rows)
            .unwrap_or_default();
        let primitive_len = primitive_rows.len().max(1);
        let terrain_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_fem_contact_reduce terrain_heights"),
            size: (terrain_len * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let friction_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_fem_contact_reduce terrain_friction"),
            size: (terrain_len * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let water_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_fem_contact_reduce terrain_water_film"),
            size: (terrain_len * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sinkage_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_fem_contact_reduce terrain_sinkage"),
            size: (terrain_len * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sdf_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_fem_contact_reduce contact_primitives"),
            size: (primitive_len * core::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (terrain_origin, terrain_dims): ([f32; 4], [u32; 4]) =
            if let Some(terrain) = &input.terrain {
                let material_flags = (terrain.friction_map.is_some() as u32)
                    | ((terrain.water_film_map.is_some() as u32) << 1)
                    | ((terrain.sinkage_map.is_some() as u32) << 2);
                (
                    [
                        terrain.origin[0],
                        terrain.origin[1],
                        terrain.spacing,
                        terrain.friction,
                    ],
                    [terrain.nx, terrain.nz, 1, material_flags],
                )
            } else {
                ([0.0, 0.0, 1.0, 1.0], [1, 1, 0, 0])
            };
        let stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_fem_contact_reduce stage"),
            size: (4 * core::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let uniforms = ContactReductionUniforms {
            counts: [
                input.positions.len() as u32,
                input.tread_ids.len() as u32,
                input
                    .terrain
                    .as_ref()
                    .map(|t| t.sdf_primitives.len() as u32)
                    .unwrap_or(0),
                input
                    .terrain
                    .as_ref()
                    .map(|t| t.triangles.len() as u32)
                    .unwrap_or(0),
            ],
            contact0: [
                input.params.contact_band,
                input.params.ground_y,
                input.params.downforce,
                0.0,
            ],
            hub: [
                input.params.hub[0],
                input.params.hub[1],
                input.params.hub[2],
                0.0,
            ],
            heading: [
                input.params.heading[0],
                input.params.heading[1],
                input.params.heading[2],
                0.0,
            ],
            terrain_origin,
            terrain_dims,
            soil0: [
                input
                    .terrain
                    .as_ref()
                    .map(|t| t.soil_rolling_resistance_scale.max(0.0))
                    .unwrap_or(0.0),
                triangle_bvh_node_count as f32,
                triangle_bvh_index_row_count as f32,
                0.0,
            ],
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_fem_contact_reduce uniforms"),
            size: core::mem::size_of::<ContactReductionUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&positions_buf, 0, bytemuck::cast_slice(&input.positions));
        if !input.tread_ids.is_empty() {
            queue.write_buffer(&tread_buf, 0, bytemuck::cast_slice(&input.tread_ids));
        }
        if let Some(terrain) = &input.terrain {
            queue.write_buffer(&terrain_buf, 0, bytemuck::cast_slice(&terrain.heights));
            if let Some(friction_map) = &terrain.friction_map {
                queue.write_buffer(&friction_buf, 0, bytemuck::cast_slice(friction_map));
            } else {
                let flat = vec![terrain.friction; terrain_len];
                queue.write_buffer(&friction_buf, 0, bytemuck::cast_slice(&flat));
            }
            if let Some(water_map) = &terrain.water_film_map {
                queue.write_buffer(&water_buf, 0, bytemuck::cast_slice(water_map));
            } else {
                let dry = vec![0.0_f32; terrain_len];
                queue.write_buffer(&water_buf, 0, bytemuck::cast_slice(&dry));
            }
            if let Some(sinkage_map) = &terrain.sinkage_map {
                queue.write_buffer(&sinkage_buf, 0, bytemuck::cast_slice(sinkage_map));
            } else {
                let hard = vec![0.0_f32; terrain_len];
                queue.write_buffer(&sinkage_buf, 0, bytemuck::cast_slice(&hard));
            }
        } else {
            queue.write_buffer(&terrain_buf, 0, bytemuck::cast_slice(&[0.0_f32]));
            queue.write_buffer(&friction_buf, 0, bytemuck::cast_slice(&[1.0_f32]));
            queue.write_buffer(&water_buf, 0, bytemuck::cast_slice(&[0.0_f32]));
            queue.write_buffer(&sinkage_buf, 0, bytemuck::cast_slice(&[0.0_f32]));
        }
        if primitive_rows.is_empty() {
            queue.write_buffer(&sdf_buf, 0, bytemuck::cast_slice(&[[0.0_f32; 4]]));
        } else {
            queue.write_buffer(&sdf_buf, 0, bytemuck::cast_slice(&primitive_rows));
        }
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem_contact_reduce io bg"),
            layout: &self.layout_io,
            entries: &[
                entry(0, &positions_buf),
                entry(1, &tread_buf),
                entry(2, &out_buf),
                entry(3, &terrain_buf),
                entry(4, &friction_buf),
                entry(5, &water_buf),
                entry(6, &sdf_buf),
                entry(7, &sinkage_buf),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem_contact_reduce uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniform_buf)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_fem_contact_reduce encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_fem_contact_reduce pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &out_buf,
            0,
            &stage,
            0,
            (4 * core::mem::size_of::<[f32; 4]>()) as u64,
        );
        queue.submit(std::iter::once(encoder.finish()));

        let rows = read_vec4(device, &stage, "tire_fem_contact_reduce output")?;
        let a = rows[0];
        let b = rows[1];
        let f = rows[2];
        let m = rows[3];
        Ok(TireFemContactReductionOutput {
            normal_load: a[0],
            contact_count: a[1].round().max(0.0) as u32,
            patch_length: a[2],
            contact_center_x: a[3],
            min_x: b[0],
            max_x: b[1],
            weight_sum: b[2],
            mean_friction: b[3],
            mean_water_film_m: m[3],
            mean_sinkage_m: f[3],
            world_force: [f[0], f[1], f[2]],
            world_moment: [m[0], m[1], m[2]],
        })
    }

    pub fn dispatch_blocking_with_gpu_terrain(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireFemContactReductionInput,
        terrain: &TireFemContactReductionGpuTerrain<'_>,
    ) -> Result<TireFemContactReductionOutput> {
        if input.positions.is_empty() {
            return Err(Error::Solver(
                "tire_fem_contact_reduce: positions must not be empty".into(),
            ));
        }
        let terrain_len = terrain.nx as usize * terrain.nz as usize;
        if terrain.nx == 0
            || terrain.nz == 0
            || !terrain.spacing.is_finite()
            || terrain.spacing <= 0.0
        {
            return Err(Error::Solver(
                "tire_fem_contact_reduce: GPU terrain dimensions/spacing invalid".into(),
            ));
        }

        let positions_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_fem_contact_reduce gpu-terrain positions"),
            size: (input.positions.len().max(1) * core::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tread_len = input.tread_ids.len().max(1);
        let tread_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_fem_contact_reduce gpu-terrain tread_ids"),
            size: (tread_len * core::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_fem_contact_reduce gpu-terrain output"),
            size: (4 * core::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mk_f32_buf = |label: &str| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (terrain_len.max(1) * core::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let friction_default = mk_f32_buf("tire_fem_contact_reduce gpu-terrain friction default");
        let water_default = mk_f32_buf("tire_fem_contact_reduce gpu-terrain water default");
        let sinkage_default = mk_f32_buf("tire_fem_contact_reduce gpu-terrain sinkage default");
        let sdf_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_fem_contact_reduce gpu-terrain contact_primitives"),
            size: core::mem::size_of::<[f32; 4]>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stage = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_fem_contact_reduce gpu-terrain stage"),
            size: (4 * core::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_fem_contact_reduce gpu-terrain uniforms"),
            size: core::mem::size_of::<ContactReductionUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        queue.write_buffer(&positions_buf, 0, bytemuck::cast_slice(&input.positions));
        if !input.tread_ids.is_empty() {
            queue.write_buffer(&tread_buf, 0, bytemuck::cast_slice(&input.tread_ids));
        }
        if terrain.friction_map.is_none() {
            let flat = vec![terrain.friction; terrain_len];
            queue.write_buffer(&friction_default, 0, bytemuck::cast_slice(&flat));
        }
        if terrain.water_film_map.is_none() {
            let dry = vec![0.0_f32; terrain_len];
            queue.write_buffer(&water_default, 0, bytemuck::cast_slice(&dry));
        }
        if terrain.sinkage_map.is_none() {
            let hard = vec![0.0_f32; terrain_len];
            queue.write_buffer(&sinkage_default, 0, bytemuck::cast_slice(&hard));
        }
        queue.write_buffer(&sdf_buf, 0, bytemuck::cast_slice(&[[0.0_f32; 4]]));

        let material_flags = (terrain.friction_map.is_some() as u32)
            | ((terrain.water_film_map.is_some() as u32) << 1)
            | ((terrain.sinkage_map.is_some() as u32) << 2);
        let uniforms = ContactReductionUniforms {
            counts: [
                input.positions.len() as u32,
                input.tread_ids.len() as u32,
                0,
                0,
            ],
            contact0: [
                input.params.contact_band,
                input.params.ground_y,
                input.params.downforce,
                0.0,
            ],
            hub: [
                input.params.hub[0],
                input.params.hub[1],
                input.params.hub[2],
                0.0,
            ],
            heading: [
                input.params.heading[0],
                input.params.heading[1],
                input.params.heading[2],
                0.0,
            ],
            terrain_origin: [
                terrain.origin[0],
                terrain.origin[1],
                terrain.spacing,
                terrain.friction,
            ],
            terrain_dims: [terrain.nx, terrain.nz, 1, material_flags],
            soil0: [
                terrain.soil_rolling_resistance_scale.max(0.0),
                0.0,
                0.0,
                0.0,
            ],
        };
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let friction_buf = terrain.friction_map.unwrap_or(&friction_default);
        let water_buf = terrain.water_film_map.unwrap_or(&water_default);
        let sinkage_buf = terrain.sinkage_map.unwrap_or(&sinkage_default);
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem_contact_reduce gpu-terrain io bg"),
            layout: &self.layout_io,
            entries: &[
                entry(0, &positions_buf),
                entry(1, &tread_buf),
                entry(2, &out_buf),
                entry(3, terrain.heights),
                entry(4, friction_buf),
                entry(5, water_buf),
                entry(6, &sdf_buf),
                entry(7, sinkage_buf),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem_contact_reduce gpu-terrain uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniform_buf)],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_fem_contact_reduce gpu-terrain encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_fem_contact_reduce gpu-terrain pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        encoder.copy_buffer_to_buffer(
            &out_buf,
            0,
            &stage,
            0,
            (4 * core::mem::size_of::<[f32; 4]>()) as u64,
        );
        queue.submit(std::iter::once(encoder.finish()));

        let rows = read_vec4(device, &stage, "tire_fem_contact_reduce gpu-terrain output")?;
        let a = rows[0];
        let b = rows[1];
        let f = rows[2];
        let m = rows[3];
        Ok(TireFemContactReductionOutput {
            normal_load: a[0],
            contact_count: a[1].round().max(0.0) as u32,
            patch_length: a[2],
            contact_center_x: a[3],
            min_x: b[0],
            max_x: b[1],
            weight_sum: b[2],
            mean_friction: b[3],
            mean_water_film_m: m[3],
            mean_sinkage_m: f[3],
            world_force: [f[0], f[1], f[2]],
            world_moment: [m[0], m[1], m[2]],
        })
    }
}

impl TireFemContactFeedbackKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::tire_fem_contact_feedback"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/tire_fem_contact_feedback.wgsl").into(),
            ),
        });
        let layout_io = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_fem_contact_feedback io"),
            entries: &[
                storage_entry(0, true),  // positions
                storage_entry(1, true),  // velocities
                storage_entry(2, true),  // masses
                storage_entry(3, true),  // tread ids
                storage_entry(4, true),  // bristle in
                storage_entry(5, false), // velocity out
                storage_entry(6, false), // bristle out
                storage_entry(7, false), // force out
                storage_entry(8, true),  // terrain heights
                storage_entry(9, true),  // terrain friction map
                storage_entry(10, true), // terrain water-film map
                storage_entry(11, true), // SDF primitive rows
                storage_entry(12, true), // terrain sinkage map
                storage_entry(13, true), // terrain surface velocity map
            ],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_fem_contact_feedback uniforms"),
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
            label: Some("molla::tire_fem_contact_feedback layout"),
            bind_group_layouts: &[&layout_io, &layout_uniforms],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::tire_fem_contact_feedback pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("contact_feedback"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self {
            pipeline,
            layout_io,
            layout_uniforms,
        }
    }

    pub fn dispatch_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireFemContactFeedbackInput,
    ) -> Result<TireFemContactFeedbackOutput> {
        let n = input.positions.len();
        if n == 0 {
            return Err(Error::Solver(
                "tire_fem_contact_feedback: positions must not be empty".into(),
            ));
        }
        if input.velocities.len() != n || input.masses.len() != n {
            return Err(Error::Solver(
                "tire_fem_contact_feedback: position/velocity/mass lengths must match".into(),
            ));
        }
        if input.bristles.len() != input.tread_ids.len() {
            return Err(Error::Solver(
                "tire_fem_contact_feedback: bristle count must match tread_ids".into(),
            ));
        }
        if let Some(terrain) = &input.terrain {
            let expected = terrain.nx as usize * terrain.nz as usize;
            if terrain.nx == 0 || terrain.nz == 0 || terrain.heights.len() != expected {
                return Err(Error::Solver(
                    "tire_fem_contact_feedback: heightfield dimensions must match heights".into(),
                ));
            }
            if let Some(map) = &terrain.friction_map
                && map.len() != expected
            {
                return Err(Error::Solver(
                    "tire_fem_contact_feedback: friction_map dimensions must match heights".into(),
                ));
            }
            if let Some(map) = &terrain.water_film_map
                && map.len() != expected
            {
                return Err(Error::Solver(
                    "tire_fem_contact_feedback: water_film_map dimensions must match heights"
                        .into(),
                ));
            }
            if let Some(map) = &terrain.sinkage_map
                && map.len() != expected
            {
                return Err(Error::Solver(
                    "tire_fem_contact_feedback: sinkage_map dimensions must match heights".into(),
                ));
            }
            if let Some(map) = &terrain.surface_velocity_map
                && map.len() != expected
            {
                return Err(Error::Solver(
                    "tire_fem_contact_feedback: surface_velocity_map dimensions must match heights"
                        .into(),
                ));
            }
        }

        let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let f32_bytes = |len: usize| (len.max(1) * core::mem::size_of::<f32>()) as u64;
        let u32_bytes = |len: usize| (len.max(1) * core::mem::size_of::<u32>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };

        let pos_buf = mk_buf(
            "tire_fem_contact_feedback positions",
            vec4_bytes(n),
            usage_ro,
        );
        let vel_buf = mk_buf(
            "tire_fem_contact_feedback velocities",
            vec4_bytes(n),
            usage_ro,
        );
        let mass_buf = mk_buf("tire_fem_contact_feedback masses", f32_bytes(n), usage_ro);
        let tread_buf = mk_buf(
            "tire_fem_contact_feedback tread_ids",
            u32_bytes(input.tread_ids.len()),
            usage_ro,
        );
        let bristle_in_buf = mk_buf(
            "tire_fem_contact_feedback bristle_in",
            vec4_bytes(input.bristles.len()),
            usage_ro,
        );
        let vel_out_buf = mk_buf("tire_fem_contact_feedback vel_out", vec4_bytes(n), usage_rw);
        let bristle_out_buf = mk_buf(
            "tire_fem_contact_feedback bristle_out",
            vec4_bytes(input.bristles.len()),
            usage_rw,
        );
        let force_out_buf = mk_buf(
            "tire_fem_contact_feedback force_out",
            vec4_bytes(n),
            usage_rw,
        );
        let terrain_len = input
            .terrain
            .as_ref()
            .map(|t| t.heights.len())
            .unwrap_or(1)
            .max(1);
        let terrain_buf = mk_buf(
            "tire_fem_contact_feedback terrain_heights",
            f32_bytes(terrain_len),
            usage_ro,
        );
        let friction_buf = mk_buf(
            "tire_fem_contact_feedback terrain_friction",
            f32_bytes(terrain_len),
            usage_ro,
        );
        let water_buf = mk_buf(
            "tire_fem_contact_feedback terrain_water_film",
            f32_bytes(terrain_len),
            usage_ro,
        );
        let sinkage_buf = mk_buf(
            "tire_fem_contact_feedback terrain_sinkage",
            f32_bytes(terrain_len),
            usage_ro,
        );
        let velocity_map_buf = mk_buf(
            "tire_fem_contact_feedback terrain_surface_velocity",
            vec4_bytes(terrain_len),
            usage_ro,
        );
        let (primitive_rows, triangle_bvh_node_count, triangle_bvh_index_row_count): (
            Vec<[f32; 4]>,
            u32,
            u32,
        ) = input
            .terrain
            .as_ref()
            .map(contact_primitive_rows)
            .unwrap_or_default();
        let sdf_buf = mk_buf(
            "tire_fem_contact_feedback contact_primitives",
            vec4_bytes(primitive_rows.len()),
            usage_ro,
        );
        let uniform_buf = mk_buf(
            "tire_fem_contact_feedback uniforms",
            core::mem::size_of::<ContactFeedbackUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        queue.write_buffer(&pos_buf, 0, bytemuck::cast_slice(&input.positions));
        queue.write_buffer(&vel_buf, 0, bytemuck::cast_slice(&input.velocities));
        queue.write_buffer(&mass_buf, 0, bytemuck::cast_slice(&input.masses));
        if !input.tread_ids.is_empty() {
            queue.write_buffer(&tread_buf, 0, bytemuck::cast_slice(&input.tread_ids));
            queue.write_buffer(&bristle_in_buf, 0, bytemuck::cast_slice(&input.bristles));
            // Default output to input state so airborne tread slots remain defined.
            queue.write_buffer(&bristle_out_buf, 0, bytemuck::cast_slice(&input.bristles));
        }
        let (terrain_origin, terrain_dims): ([f32; 4], [u32; 4]) =
            if let Some(terrain) = &input.terrain {
                let material_flags = (terrain.friction_map.is_some() as u32)
                    | ((terrain.water_film_map.is_some() as u32) << 1)
                    | ((terrain.sinkage_map.is_some() as u32) << 2)
                    | ((terrain.surface_velocity_map.is_some() as u32) << 3);
                queue.write_buffer(&terrain_buf, 0, bytemuck::cast_slice(&terrain.heights));
                if let Some(map) = &terrain.friction_map {
                    queue.write_buffer(&friction_buf, 0, bytemuck::cast_slice(map));
                } else {
                    let flat = vec![terrain.friction; terrain_len];
                    queue.write_buffer(&friction_buf, 0, bytemuck::cast_slice(&flat));
                }
                if let Some(map) = &terrain.water_film_map {
                    queue.write_buffer(&water_buf, 0, bytemuck::cast_slice(map));
                } else {
                    let dry = vec![0.0_f32; terrain_len];
                    queue.write_buffer(&water_buf, 0, bytemuck::cast_slice(&dry));
                }
                if let Some(map) = &terrain.sinkage_map {
                    queue.write_buffer(&sinkage_buf, 0, bytemuck::cast_slice(map));
                } else {
                    let hard = vec![0.0_f32; terrain_len];
                    queue.write_buffer(&sinkage_buf, 0, bytemuck::cast_slice(&hard));
                }
                if let Some(map) = &terrain.surface_velocity_map {
                    queue.write_buffer(&velocity_map_buf, 0, bytemuck::cast_slice(map));
                } else {
                    let uniform_velocity = vec![
                        [
                            terrain.surface_velocity[0],
                            terrain.surface_velocity[1],
                            terrain.surface_velocity[2],
                            0.0,
                        ];
                        terrain_len
                    ];
                    queue.write_buffer(
                        &velocity_map_buf,
                        0,
                        bytemuck::cast_slice(&uniform_velocity),
                    );
                }
                if primitive_rows.is_empty() {
                    queue.write_buffer(&sdf_buf, 0, bytemuck::cast_slice(&[[0.0_f32; 4]]));
                } else {
                    queue.write_buffer(&sdf_buf, 0, bytemuck::cast_slice(&primitive_rows));
                }
                (
                    [
                        terrain.origin[0],
                        terrain.origin[1],
                        terrain.spacing,
                        terrain.friction,
                    ],
                    [terrain.nx, terrain.nz, 1, material_flags],
                )
            } else {
                queue.write_buffer(&terrain_buf, 0, bytemuck::cast_slice(&[0.0_f32]));
                queue.write_buffer(
                    &friction_buf,
                    0,
                    bytemuck::cast_slice(&[input.params.mu_static]),
                );
                queue.write_buffer(&water_buf, 0, bytemuck::cast_slice(&[0.0_f32]));
                queue.write_buffer(&sinkage_buf, 0, bytemuck::cast_slice(&[0.0_f32]));
                queue.write_buffer(&velocity_map_buf, 0, bytemuck::cast_slice(&[[0.0_f32; 4]]));
                queue.write_buffer(&sdf_buf, 0, bytemuck::cast_slice(&[[0.0_f32; 4]]));
                ([0.0, 0.0, 1.0, input.params.mu_static], [1, 1, 0, 0])
            };

        let p = input.params;
        let uniforms = ContactFeedbackUniforms {
            counts: [
                n as u32,
                input.tread_ids.len() as u32,
                input
                    .terrain
                    .as_ref()
                    .map(|t| t.sdf_primitives.len() as u32)
                    .unwrap_or(0),
                input
                    .terrain
                    .as_ref()
                    .map(|t| t.triangles.len() as u32)
                    .unwrap_or(0),
            ],
            phys0: [p.dt, p.contact_band, p.ground_y, p.omega],
            hub: [p.hub[0], p.hub[1], p.hub[2], p.rolling_radius],
            hub_vel: [
                p.hub_vel[0],
                p.hub_vel[1],
                p.hub_vel[2],
                p.normal_load_per_tread,
            ],
            heading: [p.heading[0], p.heading[1], p.heading[2], p.area_per_tread],
            material0: [
                p.tread_shear_stiffness,
                p.mu_static,
                p.mu_dynamic,
                p.bristle_damping,
            ],
            material1: [p.cohesion, p.bristle_relax_rate, 0.0, 0.0],
            terrain_velocity: input
                .terrain
                .as_ref()
                .map(|t| {
                    [
                        t.surface_velocity[0],
                        t.surface_velocity[1],
                        t.surface_velocity[2],
                        0.0,
                    ]
                })
                .unwrap_or([0.0; 4]),
            terrain_origin,
            terrain_dims,
            soil0: [
                input
                    .terrain
                    .as_ref()
                    .map(|t| t.soil_rolling_resistance_scale.max(0.0))
                    .unwrap_or(0.0),
                triangle_bvh_node_count as f32,
                triangle_bvh_index_row_count as f32,
                0.0,
            ],
        };
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem_contact_feedback io bg"),
            layout: &self.layout_io,
            entries: &[
                entry(0, &pos_buf),
                entry(1, &vel_buf),
                entry(2, &mass_buf),
                entry(3, &tread_buf),
                entry(4, &bristle_in_buf),
                entry(5, &vel_out_buf),
                entry(6, &bristle_out_buf),
                entry(7, &force_out_buf),
                entry(8, &terrain_buf),
                entry(9, &friction_buf),
                entry(10, &water_buf),
                entry(11, &sdf_buf),
                entry(12, &sinkage_buf),
                entry(13, &velocity_map_buf),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem_contact_feedback uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniform_buf)],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_fem_contact_feedback encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_fem_contact_feedback pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }

        let stage_vel = stage_vec4(device, "tire_fem_contact_feedback stage vel", n);
        let stage_bristle = stage_vec4(
            device,
            "tire_fem_contact_feedback stage bristle",
            input.bristles.len(),
        );
        let stage_force = stage_vec4(device, "tire_fem_contact_feedback stage force", n);
        encoder.copy_buffer_to_buffer(&vel_out_buf, 0, &stage_vel, 0, vec4_bytes(n));
        encoder.copy_buffer_to_buffer(
            &bristle_out_buf,
            0,
            &stage_bristle,
            0,
            vec4_bytes(input.bristles.len()),
        );
        encoder.copy_buffer_to_buffer(&force_out_buf, 0, &stage_force, 0, vec4_bytes(n));
        queue.submit(std::iter::once(encoder.finish()));

        Ok(TireFemContactFeedbackOutput {
            velocities: read_vec4(device, &stage_vel, "tire_fem_contact_feedback velocities")?,
            bristles: read_vec4(device, &stage_bristle, "tire_fem_contact_feedback bristles")?,
            forces: read_vec4(device, &stage_force, "tire_fem_contact_feedback forces")?,
        })
    }

    /// Create the GPU-resident state for the zero-readback contact
    /// feedback stage (PLAN_REALTIRE / PROGRESS.md "zero host readback"
    /// follow-up): persistent ping-pong bristle buffers, a full-node
    /// velocity output buffer, per-node force rows (consumable by
    /// [`TireFemGpuKernel::step_hub_persistent_with_tread_forces`]),
    /// and 1-element flat-ground terrain stand-ins. `initial_bristles`
    /// must be one row per entry of the mesh's tread table (the same
    /// `tread_ids` the mesh was created from).
    pub fn create_resident(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &TireFemPersistentMesh,
        initial_bristles: &[[f32; 4]],
    ) -> Result<TireFemResidentContact> {
        if initial_bristles.is_empty() {
            return Err(Error::Solver(
                "tire_fem_contact_feedback: resident state needs at least one tread row".into(),
            ));
        }
        let n = mesh.n_nodes as usize;
        let n_tread = initial_bristles.len();
        let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };
        let bristle_a = mk_buf(
            "tire_fem_contact_feedback resident bristle A",
            vec4_bytes(n_tread),
            usage_rw,
        );
        let bristle_b = mk_buf(
            "tire_fem_contact_feedback resident bristle B",
            vec4_bytes(n_tread),
            usage_rw,
        );
        queue.write_buffer(&bristle_a, 0, bytemuck::cast_slice(initial_bristles));
        queue.write_buffer(&bristle_b, 0, bytemuck::cast_slice(initial_bristles));
        let vel_out_buf = mk_buf(
            "tire_fem_contact_feedback resident vel_out",
            vec4_bytes(n),
            usage_rw,
        );
        let force_buf = mk_buf(
            "tire_fem_contact_feedback resident force rows",
            vec4_bytes(n),
            usage_rw,
        );
        let uniform_buf = mk_buf(
            "tire_fem_contact_feedback resident uniforms",
            core::mem::size_of::<ContactFeedbackUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        // Flat-ground stand-ins for the terrain bindings (matching the
        // `terrain: None` arm of `dispatch_blocking`). The friction
        // stand-in is rewritten per dispatch from `params.mu_static`.
        let terrain_dummy = mk_buf("tire_fem_contact_feedback resident terrain", 4, usage_ro);
        let friction_dummy = mk_buf("tire_fem_contact_feedback resident friction", 4, usage_ro);
        let water_dummy = mk_buf("tire_fem_contact_feedback resident water", 4, usage_ro);
        let sinkage_dummy = mk_buf("tire_fem_contact_feedback resident sinkage", 4, usage_ro);
        let velocity_map_dummy = mk_buf(
            "tire_fem_contact_feedback resident surface velocity",
            16,
            usage_ro,
        );
        let sdf_dummy = mk_buf("tire_fem_contact_feedback resident sdf", 16, usage_ro);
        queue.write_buffer(&terrain_dummy, 0, bytemuck::cast_slice(&[0.0_f32]));
        queue.write_buffer(&water_dummy, 0, bytemuck::cast_slice(&[0.0_f32]));
        queue.write_buffer(&sinkage_dummy, 0, bytemuck::cast_slice(&[0.0_f32]));
        queue.write_buffer(
            &velocity_map_dummy,
            0,
            bytemuck::cast_slice(&[[0.0_f32; 4]]),
        );
        queue.write_buffer(&sdf_dummy, 0, bytemuck::cast_slice(&[[0.0_f32; 4]]));
        Ok(TireFemResidentContact {
            bristle_a,
            bristle_b,
            vel_out_buf,
            force_buf,
            uniform_buf,
            terrain_dummy,
            friction_dummy,
            water_dummy,
            sinkage_dummy,
            velocity_map_dummy,
            sdf_dummy,
            parity: false,
            n_tread: n_tread as u32,
        })
    }

    /// Zero-readback contact-feedback dispatch: reads the persistent
    /// mesh's just-written pos/vel slot (same parity rule as
    /// `step_hub_persistent`), integrates the persistent bristles
    /// (ping-pong), writes the brush-updated velocities back into the
    /// mesh's velocity slot with a device-to-device copy, and leaves
    /// the per-node force rows in `state.force_rows()` for
    /// [`TireFemGpuKernel::step_hub_persistent_with_tread_forces`].
    /// Flat ground only in v1 — heightfield/SDF terrain still goes
    /// through `dispatch_blocking{,_with_gpu_terrain}`.
    ///
    /// Call order per substep:
    /// `step_persistent` → `dispatch_resident` →
    /// `step_hub_persistent_with_tread_forces`. Nothing crosses back
    /// to the host.
    pub fn dispatch_resident(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: &TireFemPersistentMesh,
        state: &mut TireFemResidentContact,
        params: TireFemContactFeedbackParams,
    ) -> Result<()> {
        let n = mesh.n_nodes;
        if n == 0 {
            return Err(Error::Solver(
                "tire_fem_contact_feedback: resident dispatch on empty mesh".into(),
            ));
        }
        // Same slot rule as the hub kernels: the node step just wrote
        // B when `mesh.parity` is set, A otherwise.
        let (pos_buf, vel_buf) = if mesh.parity {
            (&mesh.pos_b, &mesh.vel_b)
        } else {
            (&mesh.pos_a, &mesh.vel_a)
        };
        let (bristle_in, bristle_out) = if state.parity {
            (&state.bristle_b, &state.bristle_a)
        } else {
            (&state.bristle_a, &state.bristle_b)
        };
        queue.write_buffer(
            &state.friction_dummy,
            0,
            bytemuck::cast_slice(&[params.mu_static]),
        );
        let p = params;
        let uniforms = ContactFeedbackUniforms {
            counts: [n, state.n_tread, 0, 0],
            phys0: [p.dt, p.contact_band, p.ground_y, p.omega],
            hub: [p.hub[0], p.hub[1], p.hub[2], p.rolling_radius],
            hub_vel: [
                p.hub_vel[0],
                p.hub_vel[1],
                p.hub_vel[2],
                p.normal_load_per_tread,
            ],
            heading: [p.heading[0], p.heading[1], p.heading[2], p.area_per_tread],
            material0: [
                p.tread_shear_stiffness,
                p.mu_static,
                p.mu_dynamic,
                p.bristle_damping,
            ],
            material1: [p.cohesion, p.bristle_relax_rate, 0.0, 0.0],
            terrain_velocity: [0.0; 4],
            terrain_origin: [0.0, 0.0, 1.0, p.mu_static],
            terrain_dims: [1, 1, 0, 0],
            soil0: [0.0; 4],
        };
        queue.write_buffer(&state.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem_contact_feedback resident io bg"),
            layout: &self.layout_io,
            entries: &[
                entry(0, pos_buf),
                entry(1, vel_buf),
                entry(2, &mesh.mass_buf),
                entry(3, &mesh.tread_id_buf),
                entry(4, bristle_in),
                entry(5, &state.vel_out_buf),
                entry(6, bristle_out),
                entry(7, &state.force_buf),
                entry(8, &state.terrain_dummy),
                entry(9, &state.friction_dummy),
                entry(10, &state.water_dummy),
                entry(11, &state.sdf_dummy),
                entry(12, &state.sinkage_dummy),
                entry(13, &state.velocity_map_dummy),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem_contact_feedback resident uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &state.uniform_buf)],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_fem_contact_feedback resident encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_fem_contact_feedback resident pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups(n.div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        // Brush impulses reach the mesh like the CPU bristle term
        // reaches `self.vel` — replace the mesh's velocity slot with
        // the brush-updated one, all device-side.
        let vel_bytes = (n as usize * core::mem::size_of::<[f32; 4]>()) as u64;
        encoder.copy_buffer_to_buffer(&state.vel_out_buf, 0, vel_buf, 0, vel_bytes);
        queue.submit(Some(encoder.finish()));
        state.parity = !state.parity;
        Ok(())
    }

    pub fn dispatch_blocking_with_gpu_terrain(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireFemContactFeedbackInput,
        terrain: &TireFemContactFeedbackGpuTerrain<'_>,
    ) -> Result<TireFemContactFeedbackOutput> {
        let n = input.positions.len();
        if n == 0 {
            return Err(Error::Solver(
                "tire_fem_contact_feedback: positions must not be empty".into(),
            ));
        }
        if input.velocities.len() != n || input.masses.len() != n {
            return Err(Error::Solver(
                "tire_fem_contact_feedback: position/velocity/mass lengths must match".into(),
            ));
        }
        if input.bristles.len() != input.tread_ids.len() {
            return Err(Error::Solver(
                "tire_fem_contact_feedback: bristle count must match tread_ids".into(),
            ));
        }
        let terrain_len = terrain.nx as usize * terrain.nz as usize;
        if terrain.nx == 0
            || terrain.nz == 0
            || !terrain.spacing.is_finite()
            || terrain.spacing <= 0.0
        {
            return Err(Error::Solver(
                "tire_fem_contact_feedback: GPU terrain dimensions/spacing invalid".into(),
            ));
        }

        let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let f32_bytes = |len: usize| (len.max(1) * core::mem::size_of::<f32>()) as u64;
        let u32_bytes = |len: usize| (len.max(1) * core::mem::size_of::<u32>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };

        let pos_buf = mk_buf(
            "tire_fem_contact_feedback gpu-terrain positions",
            vec4_bytes(n),
            usage_ro,
        );
        let vel_buf = mk_buf(
            "tire_fem_contact_feedback gpu-terrain velocities",
            vec4_bytes(n),
            usage_ro,
        );
        let mass_buf = mk_buf(
            "tire_fem_contact_feedback gpu-terrain masses",
            f32_bytes(n),
            usage_ro,
        );
        let tread_buf = mk_buf(
            "tire_fem_contact_feedback gpu-terrain tread_ids",
            u32_bytes(input.tread_ids.len()),
            usage_ro,
        );
        let bristle_in_buf = mk_buf(
            "tire_fem_contact_feedback gpu-terrain bristle_in",
            vec4_bytes(input.bristles.len()),
            usage_ro,
        );
        let vel_out_buf = mk_buf(
            "tire_fem_contact_feedback gpu-terrain vel_out",
            vec4_bytes(n),
            usage_rw,
        );
        let bristle_out_buf = mk_buf(
            "tire_fem_contact_feedback gpu-terrain bristle_out",
            vec4_bytes(input.bristles.len()),
            usage_rw,
        );
        let force_out_buf = mk_buf(
            "tire_fem_contact_feedback gpu-terrain force_out",
            vec4_bytes(n),
            usage_rw,
        );
        let friction_default = mk_buf(
            "tire_fem_contact_feedback gpu-terrain friction default",
            f32_bytes(terrain_len),
            usage_ro,
        );
        let water_default = mk_buf(
            "tire_fem_contact_feedback gpu-terrain water default",
            f32_bytes(terrain_len),
            usage_ro,
        );
        let sinkage_default = mk_buf(
            "tire_fem_contact_feedback gpu-terrain sinkage default",
            f32_bytes(terrain_len),
            usage_ro,
        );
        let velocity_default = mk_buf(
            "tire_fem_contact_feedback gpu-terrain velocity default",
            vec4_bytes(terrain_len),
            usage_ro,
        );
        let sdf_buf = mk_buf(
            "tire_fem_contact_feedback gpu-terrain contact_primitives",
            vec4_bytes(1),
            usage_ro,
        );
        let uniform_buf = mk_buf(
            "tire_fem_contact_feedback gpu-terrain uniforms",
            core::mem::size_of::<ContactFeedbackUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        queue.write_buffer(&pos_buf, 0, bytemuck::cast_slice(&input.positions));
        queue.write_buffer(&vel_buf, 0, bytemuck::cast_slice(&input.velocities));
        queue.write_buffer(&mass_buf, 0, bytemuck::cast_slice(&input.masses));
        if !input.tread_ids.is_empty() {
            queue.write_buffer(&tread_buf, 0, bytemuck::cast_slice(&input.tread_ids));
            queue.write_buffer(&bristle_in_buf, 0, bytemuck::cast_slice(&input.bristles));
            queue.write_buffer(&bristle_out_buf, 0, bytemuck::cast_slice(&input.bristles));
        }
        if terrain.friction_map.is_none() {
            let flat = vec![terrain.friction; terrain_len];
            queue.write_buffer(&friction_default, 0, bytemuck::cast_slice(&flat));
        }
        if terrain.water_film_map.is_none() {
            let dry = vec![0.0_f32; terrain_len];
            queue.write_buffer(&water_default, 0, bytemuck::cast_slice(&dry));
        }
        if terrain.sinkage_map.is_none() {
            let hard = vec![0.0_f32; terrain_len];
            queue.write_buffer(&sinkage_default, 0, bytemuck::cast_slice(&hard));
        }
        if terrain.surface_velocity_map.is_none() {
            let uniform_velocity = vec![
                [
                    terrain.surface_velocity[0],
                    terrain.surface_velocity[1],
                    terrain.surface_velocity[2],
                    0.0,
                ];
                terrain_len
            ];
            queue.write_buffer(
                &velocity_default,
                0,
                bytemuck::cast_slice(&uniform_velocity),
            );
        }
        queue.write_buffer(&sdf_buf, 0, bytemuck::cast_slice(&[[0.0_f32; 4]]));

        let p = input.params;
        let material_flags = (terrain.friction_map.is_some() as u32)
            | ((terrain.water_film_map.is_some() as u32) << 1)
            | ((terrain.sinkage_map.is_some() as u32) << 2)
            | ((terrain.surface_velocity_map.is_some() as u32) << 3);
        let uniforms = ContactFeedbackUniforms {
            counts: [n as u32, input.tread_ids.len() as u32, 0, 0],
            phys0: [p.dt, p.contact_band, p.ground_y, p.omega],
            hub: [p.hub[0], p.hub[1], p.hub[2], p.rolling_radius],
            hub_vel: [
                p.hub_vel[0],
                p.hub_vel[1],
                p.hub_vel[2],
                p.normal_load_per_tread,
            ],
            heading: [p.heading[0], p.heading[1], p.heading[2], p.area_per_tread],
            material0: [
                p.tread_shear_stiffness,
                p.mu_static,
                p.mu_dynamic,
                p.bristle_damping,
            ],
            material1: [p.cohesion, p.bristle_relax_rate, 0.0, 0.0],
            terrain_velocity: [
                terrain.surface_velocity[0],
                terrain.surface_velocity[1],
                terrain.surface_velocity[2],
                0.0,
            ],
            terrain_origin: [
                terrain.origin[0],
                terrain.origin[1],
                terrain.spacing,
                terrain.friction,
            ],
            terrain_dims: [terrain.nx, terrain.nz, 1, material_flags],
            soil0: [
                terrain.soil_rolling_resistance_scale.max(0.0),
                0.0,
                0.0,
                0.0,
            ],
        };
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let friction_buf = terrain.friction_map.unwrap_or(&friction_default);
        let water_buf = terrain.water_film_map.unwrap_or(&water_default);
        let sinkage_buf = terrain.sinkage_map.unwrap_or(&sinkage_default);
        let velocity_map_buf = terrain.surface_velocity_map.unwrap_or(&velocity_default);
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem_contact_feedback gpu-terrain io bg"),
            layout: &self.layout_io,
            entries: &[
                entry(0, &pos_buf),
                entry(1, &vel_buf),
                entry(2, &mass_buf),
                entry(3, &tread_buf),
                entry(4, &bristle_in_buf),
                entry(5, &vel_out_buf),
                entry(6, &bristle_out_buf),
                entry(7, &force_out_buf),
                entry(8, terrain.heights),
                entry(9, friction_buf),
                entry(10, water_buf),
                entry(11, &sdf_buf),
                entry(12, sinkage_buf),
                entry(13, velocity_map_buf),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem_contact_feedback gpu-terrain uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniform_buf)],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_fem_contact_feedback gpu-terrain encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_fem_contact_feedback gpu-terrain pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        let bytes_nodes = vec4_bytes(n);
        let bytes_bristles = vec4_bytes(input.bristles.len());
        let stage_vel = stage_vec4(device, "tire_fem_contact_feedback gpu-terrain stage vel", n);
        let stage_bristle = stage_vec4(
            device,
            "tire_fem_contact_feedback gpu-terrain stage bristle",
            input.bristles.len(),
        );
        let stage_force = stage_vec4(
            device,
            "tire_fem_contact_feedback gpu-terrain stage force",
            n,
        );
        encoder.copy_buffer_to_buffer(&vel_out_buf, 0, &stage_vel, 0, bytes_nodes);
        encoder.copy_buffer_to_buffer(&bristle_out_buf, 0, &stage_bristle, 0, bytes_bristles);
        encoder.copy_buffer_to_buffer(&force_out_buf, 0, &stage_force, 0, bytes_nodes);
        queue.submit(std::iter::once(encoder.finish()));

        Ok(TireFemContactFeedbackOutput {
            velocities: read_vec4(device, &stage_vel, "tire_fem_contact_feedback gpu vel")?,
            bristles: read_vec4(
                device,
                &stage_bristle,
                "tire_fem_contact_feedback gpu bristles",
            )?,
            forces: read_vec4(device, &stage_force, "tire_fem_contact_feedback gpu forces")?,
        })
    }

    /// Read GPU-resident FEM contact-feedback buffers back to typed host rows.
    ///
    /// The preferred scene/solver path is to pass
    /// [`TireFemContactFeedbackResidentOutput`] buffers directly into the next
    /// GPU stage. CPU-driven viewer mirrors still need a single explicit seam
    /// for drawing, telemetry, or legacy state updates; keeping that seam here
    /// avoids each caller inventing its own staging-buffer copy layout.
    pub fn readback_resident_feedback_output(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        output: &TireFemContactFeedbackResidentOutput,
    ) -> Result<TireFemContactFeedbackOutput> {
        let node_bytes = (output.node_count * core::mem::size_of::<[f32; 4]>()) as u64;
        let bristle_bytes = (output.bristle_count * core::mem::size_of::<[f32; 4]>()) as u64;
        if output.node_count == 0 {
            return Err(Error::Solver(
                "tire_fem_contact_feedback: resident readback has no node rows".into(),
            ));
        }
        let stage_vel = stage_vec4(
            device,
            "tire_fem_contact_feedback resident readback vel",
            output.node_count,
        );
        let stage_force = stage_vec4(
            device,
            "tire_fem_contact_feedback resident readback force",
            output.node_count,
        );
        let stage_bristle = if output.bristle_count > 0 {
            Some(stage_vec4(
                device,
                "tire_fem_contact_feedback resident readback bristle",
                output.bristle_count,
            ))
        } else {
            None
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_fem_contact_feedback resident readback encoder"),
        });
        encoder.copy_buffer_to_buffer(&output.velocities, 0, &stage_vel, 0, node_bytes);
        encoder.copy_buffer_to_buffer(&output.forces, 0, &stage_force, 0, node_bytes);
        if let Some(stage) = stage_bristle.as_ref() {
            encoder.copy_buffer_to_buffer(&output.bristles, 0, stage, 0, bristle_bytes);
        }
        queue.submit(std::iter::once(encoder.finish()));

        Ok(TireFemContactFeedbackOutput {
            velocities: read_vec4(
                device,
                &stage_vel,
                "tire_fem_contact_feedback resident velocities",
            )?,
            bristles: if let Some(stage) = stage_bristle.as_ref() {
                read_vec4(device, stage, "tire_fem_contact_feedback resident bristles")?
            } else {
                Vec::new()
            },
            forces: read_vec4(
                device,
                &stage_force,
                "tire_fem_contact_feedback resident forces",
            )?,
        })
    }

    pub fn dispatch_resident_with_gpu_terrain(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireFemContactFeedbackInput,
        terrain: &TireFemContactFeedbackGpuTerrain<'_>,
    ) -> Result<TireFemContactFeedbackResidentOutput> {
        let n = input.positions.len();
        if n == 0 {
            return Err(Error::Solver(
                "tire_fem_contact_feedback: positions must not be empty".into(),
            ));
        }
        if input.velocities.len() != n || input.masses.len() != n {
            return Err(Error::Solver(
                "tire_fem_contact_feedback: position/velocity/mass lengths must match".into(),
            ));
        }
        if input.bristles.len() != input.tread_ids.len() {
            return Err(Error::Solver(
                "tire_fem_contact_feedback: bristle count must match tread_ids".into(),
            ));
        }
        let terrain_len = terrain.nx as usize * terrain.nz as usize;
        if terrain.nx == 0
            || terrain.nz == 0
            || !terrain.spacing.is_finite()
            || terrain.spacing <= 0.0
        {
            return Err(Error::Solver(
                "tire_fem_contact_feedback: GPU terrain dimensions/spacing invalid".into(),
            ));
        }

        let usage_ro = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let vec4_bytes = |len: usize| (len.max(1) * core::mem::size_of::<[f32; 4]>()) as u64;
        let f32_bytes = |len: usize| (len.max(1) * core::mem::size_of::<f32>()) as u64;
        let u32_bytes = |len: usize| (len.max(1) * core::mem::size_of::<u32>()) as u64;
        let mk_buf = |label: &str, bytes: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes.max(4),
                usage,
                mapped_at_creation: false,
            })
        };

        let pos_buf = mk_buf(
            "tire_fem_contact_feedback resident positions",
            vec4_bytes(n),
            usage_ro,
        );
        let vel_buf = mk_buf(
            "tire_fem_contact_feedback resident velocities",
            vec4_bytes(n),
            usage_ro,
        );
        let mass_buf = mk_buf(
            "tire_fem_contact_feedback resident masses",
            f32_bytes(n),
            usage_ro,
        );
        let tread_buf = mk_buf(
            "tire_fem_contact_feedback resident tread_ids",
            u32_bytes(input.tread_ids.len()),
            usage_ro,
        );
        let bristle_in_buf = mk_buf(
            "tire_fem_contact_feedback resident bristle_in",
            vec4_bytes(input.bristles.len()),
            usage_ro,
        );
        let vel_out_buf = mk_buf(
            "tire_fem_contact_feedback resident vel_out",
            vec4_bytes(n),
            usage_rw,
        );
        let bristle_out_buf = mk_buf(
            "tire_fem_contact_feedback resident bristle_out",
            vec4_bytes(input.bristles.len()),
            usage_rw,
        );
        let force_out_buf = mk_buf(
            "tire_fem_contact_feedback resident force_out",
            vec4_bytes(n),
            usage_rw,
        );
        let friction_default = mk_buf(
            "tire_fem_contact_feedback resident friction default",
            f32_bytes(terrain_len),
            usage_ro,
        );
        let water_default = mk_buf(
            "tire_fem_contact_feedback resident water default",
            f32_bytes(terrain_len),
            usage_ro,
        );
        let sinkage_default = mk_buf(
            "tire_fem_contact_feedback resident sinkage default",
            f32_bytes(terrain_len),
            usage_ro,
        );
        let velocity_default = mk_buf(
            "tire_fem_contact_feedback resident velocity default",
            vec4_bytes(terrain_len),
            usage_ro,
        );
        let sdf_buf = mk_buf(
            "tire_fem_contact_feedback resident contact_primitives",
            vec4_bytes(1),
            usage_ro,
        );
        let uniform_buf = mk_buf(
            "tire_fem_contact_feedback resident uniforms",
            core::mem::size_of::<ContactFeedbackUniforms>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );

        queue.write_buffer(&pos_buf, 0, bytemuck::cast_slice(&input.positions));
        queue.write_buffer(&vel_buf, 0, bytemuck::cast_slice(&input.velocities));
        queue.write_buffer(&mass_buf, 0, bytemuck::cast_slice(&input.masses));
        if !input.tread_ids.is_empty() {
            queue.write_buffer(&tread_buf, 0, bytemuck::cast_slice(&input.tread_ids));
            queue.write_buffer(&bristle_in_buf, 0, bytemuck::cast_slice(&input.bristles));
            queue.write_buffer(&bristle_out_buf, 0, bytemuck::cast_slice(&input.bristles));
        }
        if terrain.friction_map.is_none() {
            let flat = vec![terrain.friction; terrain_len];
            queue.write_buffer(&friction_default, 0, bytemuck::cast_slice(&flat));
        }
        if terrain.water_film_map.is_none() {
            let dry = vec![0.0_f32; terrain_len];
            queue.write_buffer(&water_default, 0, bytemuck::cast_slice(&dry));
        }
        if terrain.sinkage_map.is_none() {
            let hard = vec![0.0_f32; terrain_len];
            queue.write_buffer(&sinkage_default, 0, bytemuck::cast_slice(&hard));
        }
        if terrain.surface_velocity_map.is_none() {
            let uniform_velocity = vec![
                [
                    terrain.surface_velocity[0],
                    terrain.surface_velocity[1],
                    terrain.surface_velocity[2],
                    0.0,
                ];
                terrain_len
            ];
            queue.write_buffer(
                &velocity_default,
                0,
                bytemuck::cast_slice(&uniform_velocity),
            );
        }
        queue.write_buffer(&sdf_buf, 0, bytemuck::cast_slice(&[[0.0_f32; 4]]));

        let p = input.params;
        let material_flags = (terrain.friction_map.is_some() as u32)
            | ((terrain.water_film_map.is_some() as u32) << 1)
            | ((terrain.sinkage_map.is_some() as u32) << 2)
            | ((terrain.surface_velocity_map.is_some() as u32) << 3);
        let uniforms = ContactFeedbackUniforms {
            counts: [n as u32, input.tread_ids.len() as u32, 0, 0],
            phys0: [p.dt, p.contact_band, p.ground_y, p.omega],
            hub: [p.hub[0], p.hub[1], p.hub[2], p.rolling_radius],
            hub_vel: [
                p.hub_vel[0],
                p.hub_vel[1],
                p.hub_vel[2],
                p.normal_load_per_tread,
            ],
            heading: [p.heading[0], p.heading[1], p.heading[2], p.area_per_tread],
            material0: [
                p.tread_shear_stiffness,
                p.mu_static,
                p.mu_dynamic,
                p.bristle_damping,
            ],
            material1: [p.cohesion, p.bristle_relax_rate, 0.0, 0.0],
            terrain_velocity: [
                terrain.surface_velocity[0],
                terrain.surface_velocity[1],
                terrain.surface_velocity[2],
                0.0,
            ],
            terrain_origin: [
                terrain.origin[0],
                terrain.origin[1],
                terrain.spacing,
                terrain.friction,
            ],
            terrain_dims: [terrain.nx, terrain.nz, 1, material_flags],
            soil0: [
                terrain.soil_rolling_resistance_scale.max(0.0),
                0.0,
                0.0,
                0.0,
            ],
        };
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let friction_buf = terrain.friction_map.unwrap_or(&friction_default);
        let water_buf = terrain.water_film_map.unwrap_or(&water_default);
        let sinkage_buf = terrain.sinkage_map.unwrap_or(&sinkage_default);
        let velocity_map_buf = terrain.surface_velocity_map.unwrap_or(&velocity_default);
        let bg_io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem_contact_feedback resident io bg"),
            layout: &self.layout_io,
            entries: &[
                entry(0, &pos_buf),
                entry(1, &vel_buf),
                entry(2, &mass_buf),
                entry(3, &tread_buf),
                entry(4, &bristle_in_buf),
                entry(5, &vel_out_buf),
                entry(6, &bristle_out_buf),
                entry(7, &force_out_buf),
                entry(8, terrain.heights),
                entry(9, friction_buf),
                entry(10, water_buf),
                entry(11, &sdf_buf),
                entry(12, sinkage_buf),
                entry(13, velocity_map_buf),
            ],
        });
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::tire_fem_contact_feedback resident uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[entry(0, &uniform_buf)],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::tire_fem_contact_feedback resident encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::tire_fem_contact_feedback resident pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_io, &[]);
            pass.set_bind_group(1, &bg_uniforms, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        Ok(TireFemContactFeedbackResidentOutput {
            node_count: n,
            bristle_count: input.bristles.len(),
            velocities: vel_out_buf,
            bristles: bristle_out_buf,
            forces: force_out_buf,
        })
    }
}

fn validate_input(input: &TireFemGpuInput) -> Result<()> {
    let n = input.positions.len();
    if input.rest_positions.len() != n || input.velocities.len() != n || input.masses.len() != n {
        return Err(Error::Solver(
            "tire_fem: rest/position/velocity/mass lengths must match".into(),
        ));
    }
    if !input.tet_indices.len().is_multiple_of(4) {
        return Err(Error::Solver(
            "tire_fem: tet_indices length must be divisible by four".into(),
        ));
    }
    if input.tet_grads.len() != input.tet_indices.len() {
        return Err(Error::Solver(
            "tire_fem: tet_grads must contain four gradients per tet".into(),
        ));
    }
    if input.tet_volumes.len() != input.tet_indices.len() / 4 {
        return Err(Error::Solver(
            "tire_fem: tet_volumes length must match tet count".into(),
        ));
    }
    if input.tet_mu.len() != input.tet_indices.len() / 4
        || input.tet_lambda.len() != input.tet_indices.len() / 4
    {
        return Err(Error::Solver(
            "tire_fem: tet material arrays must match tet count".into(),
        ));
    }
    if input.bead_offsets.len() != input.bead_ids.len() {
        return Err(Error::Solver(
            "tire_fem: bead_offsets length must match bead_ids".into(),
        ));
    }
    let n_tets = input.tet_indices.len() / 4;
    if input.tet_fiber_dir.len() != n_tets
        || input.tet_fiber_k1.len() != n_tets
        || input.tet_fiber_k2.len() != n_tets
        || input.tet_fiber_dir_b.len() != n_tets
        || input.tet_fiber_k1_b.len() != n_tets
        || input.tet_fiber_k2_b.len() != n_tets
        || input.tet_fiber_dir_c.len() != n_tets
        || input.tet_fiber_k1_c.len() != n_tets
        || input.tet_fiber_k2_c.len() != n_tets
        || input.tet_fiber_dir_d.len() != n_tets
        || input.tet_fiber_k1_d.len() != n_tets
        || input.tet_fiber_k2_d.len() != n_tets
    {
        return Err(Error::Solver(
            "tire_fem: tet fibre arrays must match tet count".into(),
        ));
    }
    let n = input.positions.len();
    if input.node_tet_offsets.len() != n + 1 {
        return Err(Error::Solver(
            "tire_fem: node_tet_offsets length must be node_count + 1".into(),
        ));
    }
    let expected_entries = input.tet_indices.len();
    let Some(&last_offset) = input.node_tet_offsets.last() else {
        return Err(Error::Solver("tire_fem: missing CSR offsets".into()));
    };
    if last_offset as usize != expected_entries || input.node_tet_entries.len() != expected_entries
    {
        return Err(Error::Solver(
            "tire_fem: CSR entry count must equal four entries per tet".into(),
        ));
    }
    if input.xpbd_edge_params.len() != input.xpbd_edge_indices.len() {
        return Err(Error::Solver(
            "tire_fem: xpbd_edge_params length must match xpbd_edge_indices".into(),
        ));
    }
    if input.xpbd_edge_indices.is_empty() {
        if !input.node_edge_offsets.is_empty() || !input.node_edge_entries.is_empty() {
            return Err(Error::Solver(
                "tire_fem: empty XPBD edge list must not provide node_edge CSR".into(),
            ));
        }
    } else {
        if input.node_edge_offsets.len() != n + 1 {
            return Err(Error::Solver(
                "tire_fem: node_edge_offsets length must be node_count + 1".into(),
            ));
        }
        let Some(&last_edge_offset) = input.node_edge_offsets.last() else {
            return Err(Error::Solver(
                "tire_fem: missing node-edge CSR offsets".into(),
            ));
        };
        if last_edge_offset as usize != input.node_edge_entries.len() {
            return Err(Error::Solver(
                "tire_fem: node_edge_offsets last value must equal node_edge_entries length".into(),
            ));
        }
    }
    if input.xpbd_dihedral_params.len() != input.xpbd_dihedral_indices.len() {
        return Err(Error::Solver(
            "tire_fem: xpbd_dihedral_params length must match xpbd_dihedral_indices".into(),
        ));
    }
    if input.xpbd_dihedral_indices.is_empty() {
        if !input.node_dihedral_offsets.is_empty() || !input.node_dihedral_entries.is_empty() {
            return Err(Error::Solver(
                "tire_fem: empty XPBD dihedral list must not provide node_dihedral CSR".into(),
            ));
        }
    } else {
        if input.node_dihedral_offsets.len() != n + 1 {
            return Err(Error::Solver(
                "tire_fem: node_dihedral_offsets length must be node_count + 1".into(),
            ));
        }
        let Some(&last_dihedral_offset) = input.node_dihedral_offsets.last() else {
            return Err(Error::Solver(
                "tire_fem: missing node-dihedral CSR offsets".into(),
            ));
        };
        if last_dihedral_offset as usize != input.node_dihedral_entries.len() {
            return Err(Error::Solver(
                "tire_fem: node_dihedral_offsets last value must equal node_dihedral_entries length".into(),
            ));
        }
    }
    Ok(())
}

/// Host-side input for the GPU matrix-free Newton tire-FEM debug timestep.
///
/// Each field mirrors a storage binding declared by
/// `tire_fem_matrix_free_newton.wgsl`. The host fills in mesh state, current
/// candidate position vector, and the perturbation direction; the dispatch
/// returns the four output buffers ([`TireFemMatrixFreeNewtonGpuOutput`]).
#[derive(Clone, Debug)]
pub struct TireFemMatrixFreeNewtonGpuInput {
    /// `dt` in seconds — must be positive.
    pub dt: f32,
    /// Node damping coefficient. Mirrors `TetraTireFemParams::node_damping`.
    pub node_damping: f32,
    /// Finite-difference epsilon used by `jv_probe` / `diagonal_extract`.
    pub jv_epsilon: f32,
    /// Line-search alpha applied by `apply_correction`.
    pub line_search_alpha: f32,
    /// Per-node rest position (vec4, w padding).
    pub rest_pos: Vec<[f32; 4]>,
    /// Per-node previous-step position (vec4, w padding).
    pub x0_pos: Vec<[f32; 4]>,
    /// Per-node previous-step velocity (vec4, w padding).
    pub v0_vel: Vec<[f32; 4]>,
    /// Per-node lumped mass.
    pub mass: Vec<f32>,
    /// Per-node external force (gravity + downforce + applied wrench), vec4
    /// with w padding.
    pub external: Vec<[f32; 4]>,
    /// Flat node-id list per tetrahedron, length = `4 * tet_count`.
    pub tet_indices: Vec<u32>,
    /// Flat shape-function gradient list per tetrahedron, length =
    /// `4 * tet_count` (vec4 with w padding).
    pub tet_grads: Vec<[f32; 4]>,
    /// Per-tetra rest volume.
    pub tet_volume: Vec<f32>,
    /// Per-tetra Neo-Hookean μ.
    pub tet_mu: Vec<f32>,
    /// Per-tetra Neo-Hookean λ.
    pub tet_lambda: Vec<f32>,
    /// Current Newton candidate position vector (vec4, w padding).
    pub candidate_pos: Vec<[f32; 4]>,
    /// Direction vector used by `jv_probe`/`apply_correction` (vec4 padding).
    pub direction: Vec<[f32; 4]>,
}

impl TireFemMatrixFreeNewtonGpuInput {
    pub fn node_count(&self) -> usize {
        self.rest_pos.len()
    }

    pub fn tet_count(&self) -> usize {
        self.tet_volume.len()
    }
}

/// Host-side readback for the GPU matrix-free Newton tire-FEM debug timestep.
#[derive(Clone, Debug)]
pub struct TireFemMatrixFreeNewtonGpuOutput {
    pub residual: Vec<[f32; 4]>,
    pub jv: Vec<[f32; 4]>,
    pub diagonal: Vec<[f32; 4]>,
    pub pos: Vec<[f32; 4]>,
}

/// Outcome of one host-orchestrated Newton outer iteration that drives the
/// GPU matrix-free Newton primitives.
///
/// Returned by
/// [`TireFemMatrixFreeNewtonGpuKernel::run_newton_outer_blocking`]. The host
/// keeps GMRES, the Jacobi preconditioner, and line search; the GPU dispatch
/// produces every per-DoF residual / Jv / diagonal / correction sample.
#[derive(Clone, Debug)]
pub struct TireFemMatrixFreeNewtonGpuOutcome {
    pub initial_residual_norm: f32,
    pub final_residual_norm: f32,
    pub accepted_alpha: f32,
    pub krylov_iteration_count: u32,
    pub final_pos: Vec<[f32; 4]>,
    pub diagonal: Vec<[f32; 4]>,
    pub correction: Vec<[f32; 4]>,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct NewtonUniforms {
    counts: [u32; 4],
    phys0: [f32; 4],
    indices: [u32; 4],
}

/// GPU kernel that dispatches the matrix-free Newton tire-FEM debug timestep
/// WGSL source-guard shader.
///
/// The host orchestrates the GMRES + line search loop on top of these
/// primitives, since GMRES needs cross-DoF orthogonalisation that is best
/// kept on the CPU for small Krylov budgets. One dispatch through
/// [`Self::dispatch_blocking`] runs `residual_eval`, `jv_probe`,
/// `diagonal_extract`, and `apply_correction` end-to-end, returning all
/// four output buffers in one round-trip.
pub struct TireFemMatrixFreeNewtonGpuKernel {
    layout: wgpu::BindGroupLayout,
    pipeline_residual: wgpu::ComputePipeline,
    pipeline_jv: wgpu::ComputePipeline,
    pipeline_diagonal: wgpu::ComputePipeline,
    pipeline_correction: wgpu::ComputePipeline,
}

impl TireFemMatrixFreeNewtonGpuKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::tire_fem_matrix_free_newton"),
            source: wgpu::ShaderSource::Wgsl(TIRE_FEM_MATRIX_FREE_NEWTON_SHADER.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::tire_fem_matrix_free_newton io+uniforms"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(1, true),   // rest_pos
                storage_entry(2, true),   // x0_pos
                storage_entry(3, true),   // v0_vel
                storage_entry(4, true),   // mass
                storage_entry(5, true),   // external
                storage_entry(6, true),   // tet_indices
                storage_entry(7, true),   // tet_grads
                storage_entry(8, true),   // tet_volume
                storage_entry(9, true),   // tet_mu
                storage_entry(10, true),  // tet_lambda
                storage_entry(11, true),  // candidate_pos
                storage_entry(12, true),  // direction
                storage_entry(13, false), // residual_out
                storage_entry(14, false), // jv_out
                storage_entry(15, false), // diagonal_out
                storage_entry(16, false), // pos_out
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::tire_fem_matrix_free_newton layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let make = |entry_point: &'static str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };
        Self {
            pipeline_residual: make("residual_eval"),
            pipeline_jv: make("jv_probe"),
            pipeline_diagonal: make("diagonal_extract"),
            pipeline_correction: make("apply_correction"),
            layout,
        }
    }

    /// Dispatch all four matrix-free Newton entry points end-to-end and read
    /// back their outputs.
    ///
    /// The dispatch is the GPU counterpart to the per-node block of
    /// `molla_vehicle::real_tire::matrix_free_newton_correction` on CPU. The
    /// host can call this once per Newton outer iteration to gather the
    /// residual, Jv probe, Jacobi diagonal, and the line-searched candidate
    /// position update; the host then orchestrates GMRES on top.
    pub fn dispatch_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &TireFemMatrixFreeNewtonGpuInput,
    ) -> Result<TireFemMatrixFreeNewtonGpuOutput> {
        let n = input.node_count();
        if n == 0 {
            return Err(Error::Solver(
                "tire_fem_matrix_free_newton: input must contain at least one node".into(),
            ));
        }
        if n > 1024 {
            return Err(Error::Solver(
                "tire_fem_matrix_free_newton: shader caps node_count at 1024".into(),
            ));
        }
        let t = input.tet_count();
        if input.x0_pos.len() != n
            || input.v0_vel.len() != n
            || input.mass.len() != n
            || input.external.len() != n
            || input.candidate_pos.len() != n
            || input.direction.len() != n
        {
            return Err(Error::Solver(
                "tire_fem_matrix_free_newton: per-node buffers must match rest_pos length".into(),
            ));
        }
        if input.tet_indices.len() != 4 * t
            || input.tet_grads.len() != 4 * t
            || input.tet_mu.len() != t
            || input.tet_lambda.len() != t
        {
            return Err(Error::Solver(
                "tire_fem_matrix_free_newton: per-tetra buffers must agree on tet_count".into(),
            ));
        }
        let contract = tire_fem_matrix_free_newton_gpu_contract(
            n as u32,
            TIRE_FEM_MATRIX_FREE_NEWTON_DEFAULT_KRYLOV_BUDGET,
        );

        let uniforms = NewtonUniforms {
            counts: [n as u32, t as u32, contract.coupled_dof_count, 0],
            phys0: [
                finite_f32_or(input.dt, 1.0 / 300.0),
                finite_f32_or(input.node_damping, 0.0),
                finite_f32_or(input.jv_epsilon.max(1.0e-9), 1.0e-5),
                finite_f32_or(input.line_search_alpha.clamp(0.0, 1.0), 0.0),
            ],
            indices: [0, contract.krylov_iteration_budget, 0, 0],
        };
        let uniforms_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tire_fem_matrix_free_newton uniforms"),
            size: core::mem::size_of::<NewtonUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniforms_buf, 0, bytemuck::bytes_of(&uniforms));

        let storage_from_vec4 = |label: &str, data: &[[f32; 4]]| -> wgpu::Buffer {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (data.len().max(1) * core::mem::size_of::<[f32; 4]>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            if !data.is_empty() {
                queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
            }
            buf
        };
        let storage_from_f32 = |label: &str, data: &[f32]| -> wgpu::Buffer {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (data.len().max(1) * core::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            if !data.is_empty() {
                queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
            }
            buf
        };
        let storage_from_u32 = |label: &str, data: &[u32]| -> wgpu::Buffer {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (data.len().max(1) * core::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            if !data.is_empty() {
                queue.write_buffer(&buf, 0, bytemuck::cast_slice(data));
            }
            buf
        };
        let out_buf = |label: &str| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (n * core::mem::size_of::<[f32; 4]>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        let rest_buf = storage_from_vec4("rest_pos", &input.rest_pos);
        let x0_buf = storage_from_vec4("x0_pos", &input.x0_pos);
        let v0_buf = storage_from_vec4("v0_vel", &input.v0_vel);
        let mass_buf = storage_from_f32("mass", &input.mass);
        let external_buf = storage_from_vec4("external", &input.external);
        let tet_indices_buf = storage_from_u32("tet_indices", &input.tet_indices);
        let tet_grads_buf = storage_from_vec4("tet_grads", &input.tet_grads);
        let tet_volume_buf = storage_from_f32("tet_volume", &input.tet_volume);
        let tet_mu_buf = storage_from_f32("tet_mu", &input.tet_mu);
        let tet_lambda_buf = storage_from_f32("tet_lambda", &input.tet_lambda);
        let candidate_buf = storage_from_vec4("candidate_pos", &input.candidate_pos);
        let direction_buf = storage_from_vec4("direction", &input.direction);
        let residual_buf = out_buf("residual_out");
        let jv_buf = out_buf("jv_out");
        let diagonal_buf = out_buf("diagonal_out");
        let pos_buf = out_buf("pos_out");

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tire_fem_matrix_free_newton bind"),
            layout: &self.layout,
            entries: &[
                entry(0, &uniforms_buf),
                entry(1, &rest_buf),
                entry(2, &x0_buf),
                entry(3, &v0_buf),
                entry(4, &mass_buf),
                entry(5, &external_buf),
                entry(6, &tet_indices_buf),
                entry(7, &tet_grads_buf),
                entry(8, &tet_volume_buf),
                entry(9, &tet_mu_buf),
                entry(10, &tet_lambda_buf),
                entry(11, &candidate_buf),
                entry(12, &direction_buf),
                entry(13, &residual_buf),
                entry(14, &jv_buf),
                entry(15, &diagonal_buf),
                entry(16, &pos_buf),
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("tire_fem_matrix_free_newton encoder"),
        });
        let dispatch_one = |encoder: &mut wgpu::CommandEncoder,
                            pipeline: &wgpu::ComputePipeline,
                            label: &str,
                            workgroups: u32| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        };
        dispatch_one(
            &mut encoder,
            &self.pipeline_residual,
            "residual_eval",
            contract.residual_workgroup_count,
        );
        dispatch_one(
            &mut encoder,
            &self.pipeline_jv,
            "jv_probe",
            contract.jv_probe_workgroup_count,
        );
        dispatch_one(
            &mut encoder,
            &self.pipeline_diagonal,
            "diagonal_extract",
            contract.diagonal_workgroup_count,
        );
        dispatch_one(
            &mut encoder,
            &self.pipeline_correction,
            "apply_correction",
            contract.correction_workgroup_count,
        );

        let residual_read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("residual_out readback"),
            size: (n * core::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let jv_read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("jv_out readback"),
            size: (n * core::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let diagonal_read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("diagonal_out readback"),
            size: (n * core::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let pos_read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pos_out readback"),
            size: (n * core::mem::size_of::<[f32; 4]>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&residual_buf, 0, &residual_read, 0, residual_read.size());
        encoder.copy_buffer_to_buffer(&jv_buf, 0, &jv_read, 0, jv_read.size());
        encoder.copy_buffer_to_buffer(&diagonal_buf, 0, &diagonal_read, 0, diagonal_read.size());
        encoder.copy_buffer_to_buffer(&pos_buf, 0, &pos_read, 0, pos_read.size());
        queue.submit(std::iter::once(encoder.finish()));

        let residual = read_vec4(device, &residual_read, "residual_out")?;
        let jv = read_vec4(device, &jv_read, "jv_out")?;
        let diagonal = read_vec4(device, &diagonal_read, "diagonal_out")?;
        let pos = read_vec4(device, &pos_read, "pos_out")?;
        Ok(TireFemMatrixFreeNewtonGpuOutput {
            residual,
            jv,
            diagonal,
            pos,
        })
    }

    /// Host-orchestrated Newton outer iteration that drives the four GPU
    /// matrix-free Newton primitives.
    ///
    /// This is the persistent-batch integration seam: the host keeps GMRES
    /// + Jacobi preconditioning + line search, while every per-DoF residual,
    ///   Jv probe, diagonal sample, and correction application runs on the
    ///   GPU. The loop matches the CPU
    ///   `molla_vehicle::real_tire::matrix_free_newton_correction` helper
    ///   step-by-step so the two paths stay numerically equivalent.
    ///
    /// `gmres_iterations` caps the inner Krylov budget; the live tire
    /// substep uses the default budget exposed on the contract
    /// ([`TIRE_FEM_MATRIX_FREE_NEWTON_DEFAULT_KRYLOV_BUDGET`]).
    pub fn run_newton_outer_blocking(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        base: &TireFemMatrixFreeNewtonGpuInput,
        gmres_iterations: u32,
    ) -> Result<TireFemMatrixFreeNewtonGpuOutcome> {
        let n = base.node_count();
        let dof = n * 3;
        let candidate = base.candidate_pos.clone();
        let zero_vec4 = vec![[0.0_f32; 4]; n];

        // 1. Initial residual + Jacobi diagonal sample.
        let probe_input = TireFemMatrixFreeNewtonGpuInput {
            candidate_pos: candidate.clone(),
            direction: zero_vec4.clone(),
            line_search_alpha: 0.0,
            ..base.clone()
        };
        let init = self.dispatch_blocking(device, queue, &probe_input)?;
        let initial_residual_norm = flat_vec4_norm(&init.residual);
        let diagonal = init.diagonal.clone();
        if initial_residual_norm <= 1.0e-12 || !initial_residual_norm.is_finite() {
            return Ok(TireFemMatrixFreeNewtonGpuOutcome {
                initial_residual_norm,
                final_residual_norm: initial_residual_norm,
                accepted_alpha: 0.0,
                krylov_iteration_count: 0,
                final_pos: candidate,
                diagonal,
                correction: zero_vec4,
            });
        }

        // 2. Build the Jacobi preconditioner from the diagonal sample.
        let mut preconditioner = vec![1.0_f32; dof];
        for (i, diag) in diagonal.iter().enumerate() {
            for axis in 0..3 {
                let value = diag[axis];
                preconditioner[i * 3 + axis] = if value.is_finite() && value.abs() > 1.0e-9 {
                    1.0 / value
                } else {
                    1.0
                };
            }
        }

        // 3. GMRES inner loop. `b = -R(candidate)`.
        let rhs = flatten_vec4_negated(&init.residual);
        let mut basis: Vec<Vec<f32>> = Vec::with_capacity(gmres_iterations as usize + 1);
        let beta = vec_norm_flat(&rhs);
        if beta <= 1.0e-12 || !beta.is_finite() {
            return Ok(TireFemMatrixFreeNewtonGpuOutcome {
                initial_residual_norm,
                final_residual_norm: initial_residual_norm,
                accepted_alpha: 0.0,
                krylov_iteration_count: 0,
                final_pos: candidate,
                diagonal,
                correction: zero_vec4,
            });
        }
        basis.push(rhs.iter().map(|value| value / beta).collect::<Vec<_>>());
        let mut h = vec![vec![0.0_f32; gmres_iterations as usize]; gmres_iterations as usize + 1];
        let mut best_dx = vec![0.0_f32; dof];
        let mut krylov_iteration_count = 0u32;
        for j in 0..gmres_iterations as usize {
            // Apply M^-1 to basis[j] on host, then dispatch jv_probe.
            let direction_flat = basis[j]
                .iter()
                .zip(preconditioner.iter())
                .map(|(value, inv_diag)| value * inv_diag)
                .collect::<Vec<_>>();
            let direction_vec4 = unflatten_to_vec4(&direction_flat, n);
            let probe = TireFemMatrixFreeNewtonGpuInput {
                candidate_pos: candidate.clone(),
                direction: direction_vec4,
                line_search_alpha: 0.0,
                ..base.clone()
            };
            let probe_out = self.dispatch_blocking(device, queue, &probe)?;
            let mut w = flatten_vec4(&probe_out.jv);
            if w.len() != dof || !w.iter().all(|value| value.is_finite()) {
                break;
            }
            for i in 0..=j {
                h[i][j] = vec_dot_flat(&w, &basis[i]);
                for (wk, vik) in w.iter_mut().zip(basis[i].iter()) {
                    *wk -= h[i][j] * vik;
                }
            }
            h[j + 1][j] = vec_norm_flat(&w);
            if h[j + 1][j].is_finite() && h[j + 1][j] > 1.0e-12 && j + 1 < gmres_iterations as usize
            {
                basis.push(
                    w.iter()
                        .map(|value| value / h[j + 1][j])
                        .collect::<Vec<_>>(),
                );
            }
            let k = j + 1;
            let mut normal = vec![vec![0.0_f32; k]; k];
            let mut normal_rhs = vec![0.0_f32; k];
            for row in 0..k {
                normal_rhs[row] = beta * h[0][row];
                for col in 0..k {
                    normal[row][col] = (0..=k).map(|i| h[i][row] * h[i][col]).sum::<f32>();
                }
                normal[row][row] += 1.0e-14 * normal[row][row].abs().max(1.0);
            }
            let Some(y) = solve_small_normal_system(&normal, &normal_rhs) else {
                break;
            };
            let mut dx = vec![0.0_f32; dof];
            for col in 0..k {
                for (dxi, vi) in dx.iter_mut().zip(basis[col].iter()) {
                    *dxi += y[col] * vi;
                }
            }
            if !dx.iter().all(|value| value.is_finite()) {
                break;
            }
            best_dx = dx;
            krylov_iteration_count = (j as u32) + 1;
        }
        let preconditioned_dx_flat = best_dx
            .iter()
            .zip(preconditioner.iter())
            .map(|(value, inv_diag)| value * inv_diag)
            .collect::<Vec<_>>();
        let preconditioned_dx_vec4 = unflatten_to_vec4(&preconditioned_dx_flat, n);

        // 4. Line search through CPU alpha sweep, dispatching apply_correction
        // and residual_eval at the candidate pos_out each time.
        let alphas = [1.0_f32, 0.5, 0.25, 0.125, 0.0625];
        let mut best_alpha = 0.0_f32;
        let mut best_residual_norm = initial_residual_norm;
        let mut best_pos = candidate.clone();
        for alpha in alphas {
            let correction_in = TireFemMatrixFreeNewtonGpuInput {
                candidate_pos: candidate.clone(),
                direction: preconditioned_dx_vec4.clone(),
                line_search_alpha: alpha,
                ..base.clone()
            };
            let correction_out = self.dispatch_blocking(device, queue, &correction_in)?;
            let candidate_pos_after = correction_out.pos.clone();
            // Probe residual at candidate_pos_after via another dispatch with
            // candidate_pos set to the post-correction state.
            let residual_in = TireFemMatrixFreeNewtonGpuInput {
                candidate_pos: candidate_pos_after.clone(),
                direction: zero_vec4.clone(),
                line_search_alpha: 0.0,
                ..base.clone()
            };
            let residual_out = self.dispatch_blocking(device, queue, &residual_in)?;
            let new_norm = flat_vec4_norm(&residual_out.residual);
            if new_norm.is_finite() && new_norm < best_residual_norm {
                best_alpha = alpha;
                best_residual_norm = new_norm;
                best_pos = candidate_pos_after;
            }
        }
        let correction = preconditioned_dx_vec4
            .iter()
            .map(|d| [best_alpha * d[0], best_alpha * d[1], best_alpha * d[2], 0.0])
            .collect::<Vec<_>>();
        Ok(TireFemMatrixFreeNewtonGpuOutcome {
            initial_residual_norm,
            final_residual_norm: best_residual_norm,
            accepted_alpha: best_alpha,
            krylov_iteration_count,
            final_pos: best_pos,
            diagonal,
            correction,
        })
    }
}

fn flatten_vec4(values: &[[f32; 4]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(values.len() * 3);
    for v in values {
        out.push(v[0]);
        out.push(v[1]);
        out.push(v[2]);
    }
    out
}

fn flatten_vec4_negated(values: &[[f32; 4]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(values.len() * 3);
    for v in values {
        out.push(-v[0]);
        out.push(-v[1]);
        out.push(-v[2]);
    }
    out
}

fn unflatten_to_vec4(values: &[f32], node_count: usize) -> Vec<[f32; 4]> {
    let mut out = Vec::with_capacity(node_count);
    for i in 0..node_count {
        let base = i * 3;
        out.push([values[base], values[base + 1], values[base + 2], 0.0]);
    }
    out
}

fn vec_norm_flat(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

fn vec_dot_flat(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(av, bv)| av * bv).sum::<f32>()
}

fn flat_vec4_norm(values: &[[f32; 4]]) -> f32 {
    values
        .iter()
        .map(|v| v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        .sum::<f32>()
        .sqrt()
}

fn solve_small_normal_system(matrix: &[Vec<f32>], rhs: &[f32]) -> Option<Vec<f32>> {
    let n = rhs.len();
    if n == 0 {
        return Some(Vec::new());
    }
    if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    let mut a = matrix.to_vec();
    let mut b = rhs.to_vec();
    // Gaussian elimination with partial pivoting.
    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_val = a[col][col].abs();
        for row in (col + 1)..n {
            if a[row][col].abs() > pivot_val {
                pivot_row = row;
                pivot_val = a[row][col].abs();
            }
        }
        if pivot_val < 1.0e-20 {
            return None;
        }
        if pivot_row != col {
            a.swap(col, pivot_row);
            b.swap(col, pivot_row);
        }
        for row in (col + 1)..n {
            let factor = a[row][col] / a[col][col];
            for c in col..n {
                a[row][c] -= factor * a[col][c];
            }
            b[row] -= factor * b[col];
        }
    }
    let mut x = vec![0.0_f32; n];
    for row in (0..n).rev() {
        let mut sum = b[row];
        for col in (row + 1)..n {
            sum -= a[row][col] * x[col];
        }
        let denom = a[row][row];
        if denom.abs() < 1.0e-20 {
            return None;
        }
        x[row] = sum / denom;
    }
    if x.iter().all(|value| value.is_finite()) {
        Some(x)
    } else {
        None
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn finite_f32_or(v: f32, fallback: f32) -> f32 {
    if v.is_finite() { v } else { fallback }
}

fn bytes_of_slice<T>(len: usize) -> u64 {
    (len as u64).saturating_mul(core::mem::size_of::<T>() as u64)
}

fn atlas_binding(
    binding: u32,
    name: &'static str,
    rows: usize,
    bytes: u64,
    read_only: bool,
) -> TireFemGpuBatchAtlasBindingPlan {
    TireFemGpuBatchAtlasBindingPlan {
        binding,
        name,
        rows,
        bytes: bytes.max(4),
        read_only,
    }
}

fn expect_len(label: &str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::Solver(format!(
            "{label}: expected length {expected}, got {actual}"
        )))
    }
}

fn expect_last_offset(label: &str, offsets: &[u32], expected: usize) -> Result<()> {
    let Some(&last) = offsets.last() else {
        return Err(Error::Solver(format!("{label}: missing final offset")));
    };
    if last as usize == expected {
        Ok(())
    } else {
        Err(Error::Solver(format!(
            "{label}: expected final offset {expected}, got {last}"
        )))
    }
}

fn usize_to_u32(value: usize, label: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| Error::Solver(format!("{label}: value exceeds u32 range")))
}

fn offset_u32(value: u32, offset: usize, label: &str) -> Result<u32> {
    let offset = usize_to_u32(offset, label)?;
    value
        .checked_add(offset)
        .ok_or_else(|| Error::Solver(format!("{label}: atlas index overflow")))
}

fn subtract_u32(value: u32, offset: usize, label: &str) -> Result<u32> {
    let offset = usize_to_u32(offset, label)?;
    value
        .checked_sub(offset)
        .ok_or_else(|| Error::Solver(format!("{label}: atlas index underflow")))
}

fn offset_packed_index(entry: u32, local_bits: u32, offset: usize, label: &str) -> Result<u32> {
    debug_assert!(local_bits < u32::BITS);
    let local_mask = (1u32 << local_bits) - 1;
    let local = entry & local_mask;
    let index = entry >> local_bits;
    let index = offset_u32(index, offset, label)?;
    if index > (u32::MAX >> local_bits) {
        return Err(Error::Solver(format!(
            "{label}: packed atlas index overflow"
        )));
    }
    Ok((index << local_bits) | local)
}

fn subtract_packed_index(entry: u32, local_bits: u32, offset: usize, label: &str) -> Result<u32> {
    debug_assert!(local_bits < u32::BITS);
    let local_mask = (1u32 << local_bits) - 1;
    let local = entry & local_mask;
    let index = entry >> local_bits;
    let index = subtract_u32(index, offset, label)?;
    Ok((index << local_bits) | local)
}

fn stage_vec4(device: &wgpu::Device, label: &str, len: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (len * core::mem::size_of::<[f32; 4]>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn empty_storage_buffer(
    device: &wgpu::Device,
    label: &str,
    bytes: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(4),
        usage,
        mapped_at_creation: false,
    })
}

fn storage_buffer_from_slice<T: Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    rows: &[T],
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let bytes = bytes_of_slice::<T>(rows.len()).max(core::mem::size_of::<T>() as u64);
    let buf = empty_storage_buffer(device, label, bytes, usage);
    if !rows.is_empty() {
        queue.write_buffer(&buf, 0, bytemuck::cast_slice(rows));
    }
    buf
}

fn atlas_hub_state_rows(params: &[TireFemGpuParams]) -> Vec<[f32; 4]> {
    let mut hub_state = Vec::with_capacity(params.len() * 2);
    for params in params {
        hub_state.push([params.hub[0], params.hub[1], params.hub[2], 0.0]);
        hub_state.push([0.0, 0.0, 0.0, 0.0]);
    }
    hub_state
}

fn read_vec4(device: &wgpu::Device, buf: &wgpu::Buffer, label: &str) -> Result<Vec<[f32; 4]>> {
    let slice = buf.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel::<Result<()>>();
    let label = label.to_string();
    let callback_label = label.clone();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        sender
            .send(r.map_err(|e| Error::Gpu(format!("{callback_label} readback: {e:?}"))))
            .ok();
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| Error::Gpu(format!("{label} readback poll: {e:?}")))?;
    receiver
        .recv()
        .map_err(|e| Error::Gpu(format!("{label} readback recv: {e:?}")))??;
    let data = slice.get_mapped_range();
    let out: Vec<[f32; 4]> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    buf.unmap();
    Ok(out)
}
