//! Volumetric tetra FEM tire state types: `StructuralEdge`,
//! `SidewallDihedral`, `TetraTireFem`, `TetraTireFemGpuSnapshot`.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use super::super::TetraTireFemParams;

#[derive(Clone, Copy, Debug)]
pub(crate) struct StructuralEdge {
    pub(crate) ids: [usize; 2],
    pub(crate) rest_length: f64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SidewallDihedral {
    /// Quad vertices `(a,b,c,d)` around one sidewall surface cell. The
    /// diagnostic measures the fold angle between triangles `(a,b,c)` and
    /// `(a,c,d)` across diagonal `a-c`.
    pub(crate) ids: [usize; 4],
    pub(crate) rest_angle_rad: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct TetraElement {
    pub(crate) ids: [usize; 4],
    pub(crate) grads: [[f64; 3]; 4],
    pub(crate) volume: f64,
    pub(crate) mu: f64,
    pub(crate) lambda: f64,
    /// Stage (c) — first ply/fibre family in the rest configuration
    /// (unit vector). Tread/belt and sidewall regions use crossed
    /// `fiber_dir_a`/`fiber_dir_b` layups plus stage-(c3) cap/radial
    /// families; pure-rubber regions have zero stiffness and collapse to
    /// standard Neo-Hookean.
    pub(crate) fiber_dir: [f64; 3],
    /// Stage (c) — fibre stiffness k₁ (Pa). The anisotropic term in
    /// `step_inner` is `2 · k₁ · (I₄ − 1) · exp(k₂·(I₄−1)²) · F·(a₀⊗a₀)`
    /// for `I₄ > 1`. Cord/belt fibres buckle in compression so the
    /// stress collapses to zero there.
    pub(crate) fiber_k1: f64,
    /// Stage (c) — Holzapfel exponential exponent k₂ (dimensionless).
    /// With `k₂ > 0` the tangent stiffness rises with pre-stretch, so a
    /// cavity-pressure-pretensioned belt becomes progressively stiffer
    /// under further deformation. `0` collapses to a linear fibre.
    pub(crate) fiber_k2: f64,
    /// Stage (c2) — second crossed ply/fibre family for explicit ± belt /
    /// sidewall layups.
    pub(crate) fiber_dir_b: [f64; 3],
    pub(crate) fiber_k1_b: f64,
    pub(crate) fiber_k2_b: f64,
    /// Stage (c3) — third ply family for richer multi-ply tire stacks
    /// (circumferential cap/hoop ply in tread and bead, radial ply in
    /// sidewall).
    pub(crate) fiber_dir_c: [f64; 3],
    pub(crate) fiber_k1_c: f64,
    pub(crate) fiber_k2_c: f64,
    /// Stage (c3) — fourth ply family for axial/sidewall balance. This is
    /// still a compact explicit fibre stack, not a laminated shell/dihedral
    /// sidewall solve, but it gives the FEM material more than one crossed
    /// pair.
    pub(crate) fiber_dir_d: [f64; 3],
    pub(crate) fiber_k1_d: f64,
    pub(crate) fiber_k2_d: f64,
}

/// Runtime state for the volumetric tetrahedral tire FEM.
#[derive(Clone, Debug)]
pub struct TetraTireFem {
    pub(crate) params: TetraTireFemParams,
    pub(crate) hub: [f64; 3],
    pub(crate) hub_v_y: f64,
    /// Stage (f) ext — lateral / longitudinal hub velocity (m/s).
    /// Together with `hub_v_y` and `hub` gives the 3-DOF hub state.
    pub(crate) hub_v_xz: [f64; 2],
    /// PLAN_TOURQUE Stage A — spin angle around the wheel's spin axis
    /// (radians). Sign convention: positive `hub_angle` rotates the tread
    /// nodes the same way a positive `hub_omega` does. Stage A only
    /// stores it; Stage B integrates it from torque; Stage C uses it to
    /// drive contact-patch tangential velocity.
    pub(crate) hub_angle: f64,
    /// PLAN_TOURQUE Stage A — spin angular velocity (rad/s) around the
    /// wheel's spin axis. Stage B integrates this from drive/brake
    /// torque; Stage C/D couple it to contact friction.
    pub(crate) hub_omega: f64,
    /// PLAN_TOURQUE Stage A — commanded drive torque at the wheel hub
    /// (N·m, signed). Positive accelerates `hub_omega` toward +ω.
    pub(crate) drive_torque_n_m: f64,
    /// PLAN_TOURQUE Stage A — commanded brake torque at the wheel hub
    /// (N·m, ≥0). Always opposes the current `hub_omega`.
    pub(crate) brake_torque_n_m: f64,
    /// PLAN_TOURQUE Stage A — last step's net wrench at the hub (N).
    /// Stage D writes the integrated contact reaction here so a future
    /// chassis can pull it out via `last_hub_reaction_xyz`.
    pub(crate) hub_reaction_xyz: [f64; 3],
    /// PLAN_TOURQUE Stage D — when `true`, the hub XZ pose is whatever
    /// the host last passed to `set_hub_kinematic` (test-bench style:
    /// the simulator drives the hub on a predetermined path). When
    /// `false`, `hub_v_xz` integrates from the per-substep net contact
    /// reaction and the hub XZ position advances accordingly (the
    /// wheel rolls forward as an emergent consequence of contact
    /// friction). Defaults to `true` so every existing scene keeps
    /// its current behaviour byte-for-byte until it opts in.
    pub(crate) hub_kinematic: bool,
    /// Stage (f) ext — external hub force (N) — controllers (steering /
    /// drive / lateral pull) can set this each frame; it is forwarded
    /// to the GPU hub kernel and to the CPU integrator.
    pub external_force_xyz: [f64; 3],
    pub(crate) rest: Vec<[f64; 3]>,
    pub(crate) pos: Vec<[f64; 3]>,
    pub(crate) vel: Vec<[f64; 3]>,
    /// PLAN_ROLL Stage A — body-frame node positions. The world-frame
    /// `pos` vector is `R(hub_angle) · body_pos + hub`. Stage A only
    /// stores body state; Stages B/C/D wire it into the per-substep
    /// physics and contact transform. With `hub_angle = 0` the body
    /// and world frames coincide modulo `hub` translation.
    pub(crate) body_pos: Vec<[f64; 3]>,
    /// PLAN_ROLL Stage A — body-frame node velocities. Mirrors
    /// `body_pos`; same staging story.
    pub(crate) body_vel: Vec<[f64; 3]>,
    pub(crate) mass: Vec<f64>,
    pub(crate) tets: Vec<TetraElement>,
    pub(crate) surface_edges: Vec<(usize, usize)>,
    pub(crate) tread_belt_edges: Vec<StructuralEdge>,
    pub(crate) sidewall_bend_edges: Vec<StructuralEdge>,
    pub(crate) sidewall_dihedrals: Vec<SidewallDihedral>,
    pub(crate) tread_nodes: Vec<usize>,
    pub(crate) bead_offsets: Vec<(usize, [f64; 3])>,
    /// Inner-cavity surface — bead-layer (`r=0`) triangles, plus the two
    /// virtual axial end-caps. See `cavity_volume` and the gas-law
    /// pressure traction in `step_inner`.
    pub(crate) cavity_tube_tris: Vec<[u32; 3]>,
    pub(crate) cavity_neg_edges: Vec<[u32; 2]>, // pairs with a virtual axis vertex at -width/2
    pub(crate) cavity_pos_edges: Vec<[u32; 2]>, // pairs with a virtual axis vertex at +width/2
    pub(crate) cavity_volume_rest: f64,
    /// Stage (d) — per-tread-node bristle (shear-displacement vector).
    /// Indexed by tread index, *not* node index — use `node_to_tread`
    /// to map from a node id.
    pub(crate) tread_bristle: Vec<[f64; 3]>,
    /// Map node-id → tread-index, or `None` if the node is not a tread
    /// node. Built once in `new()`.
    pub(crate) node_to_tread: Vec<Option<u32>>,
    /// Contact area per tread node, in m². Used to scale the brush
    /// stiffness and material friction caps. Precomputed in `new()`.
    pub(crate) tread_area_per_node: f64,
    pub(crate) last_xpbd_volume_correction_m: f64,
    pub(crate) last_xpbd_bead_correction_m: f64,
    pub(crate) last_xpbd_structural_correction_m: f64,
    /// PLAN_REALTIRE Stream B Stage C.6 — persistent per-substep scratch
    /// buffers so `step_inner` does not re-allocate every call.
    /// Sized to `pos.len()`; cleared (`fill(...)`) at the top of each
    /// substep instead of re-allocated.
    pub(crate) scratch_force: Vec<[f64; 3]>,
    pub(crate) scratch_node_stiffness_diag: Vec<f64>,
}

/// Raw f32 mesh/state payload for GPU tire FEM kernels.
#[derive(Clone, Debug)]
pub struct TetraTireFemGpuSnapshot {
    pub rest_positions: Vec<[f32; 4]>,
    pub positions: Vec<[f32; 4]>,
    pub velocities: Vec<[f32; 4]>,
    pub masses: Vec<f32>,
    pub tet_indices: Vec<u32>,
    pub tet_grads: Vec<[f32; 4]>,
    pub tet_volumes: Vec<f32>,
    pub tet_mu: Vec<f32>,
    pub tet_lambda: Vec<f32>,
    /// Stage (c) — per-tet fibre direction `a₀` in rest config (xyz, w=0).
    pub tet_fiber_dir: Vec<[f32; 4]>,
    /// Stage (c) — per-tet linear fibre stiffness k₁ (Pa).
    pub tet_fiber_k1: Vec<f32>,
    /// Stage (c) — per-tet Holzapfel exponent k₂ (dim-less).
    pub tet_fiber_k2: Vec<f32>,
    /// Stage (c2) — second explicit crossed ply direction.
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
    pub pressure_area_per_tread_node: f32,
    /// Stage (b) — pre-computed cavity-pressure radial force *per bead
    /// node* (N). The CPU evaluates the polytropic gas law from the
    /// current `cavity_volume()` and the rest volume, then divides the
    /// resulting bead-surface load equally across all bead nodes. The
    /// GPU kernel applies this along `(node − hub)` direction.
    pub cavity_radial_force_per_bead: f32,
    /// Stage (h) CSR — for each node `i`, `node_tet_entries[node_tet_offsets[i]..node_tet_offsets[i+1]]`
    /// gives the packed `(tet_index << 2) | local_vertex_id` for every
    /// tet incident on it. Replaces the O(nodes × tets) reference loop.
    pub node_tet_offsets: Vec<u32>,
    pub node_tet_entries: Vec<u32>,
    /// Stage (f) GPU XPBD structural edges. `xpbd_edge_indices[i] = (a,b,kind,0)`;
    /// `xpbd_edge_params[i] = (rest_length, compliance, 0, 0)`.
    pub xpbd_edge_indices: Vec<[u32; 4]>,
    pub xpbd_edge_params: Vec<[f32; 4]>,
    /// Stage (f) CSR adjacency for structural edges. Entry packs
    /// `(edge_index << 1) | local_endpoint`.
    pub node_edge_offsets: Vec<u32>,
    pub node_edge_entries: Vec<u32>,
    /// Stage (f3) GPU XPBD sidewall dihedral constraints.
    /// `xpbd_dihedral_indices[i] = (a,b,c,d)` and
    /// `xpbd_dihedral_params[i] = (rest_angle_rad, compliance, 0, 0)`.
    pub xpbd_dihedral_indices: Vec<[u32; 4]>,
    pub xpbd_dihedral_params: Vec<[f32; 4]>,
    /// Stage (f3) CSR adjacency for sidewall dihedrals. Entry packs
    /// `(dihedral_index << 2) | local_vertex`.
    pub node_dihedral_offsets: Vec<u32>,
    pub node_dihedral_entries: Vec<u32>,
}

pub(crate) fn tire_nodes_finite(tire: &TetraTireFem) -> bool {
    tire.nodes().iter().all(|p| p.iter().all(|v| v.is_finite()))
        && tire.hub().iter().all(|v| v.is_finite())
        && tire.max_node_speed_m_per_s().is_finite()
}
