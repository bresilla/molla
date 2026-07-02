//! `impl TetraTireFem`: constructors, FEM physics, contact, XPBD solver.
//!
//! Moved out of `real_tire/mod.rs` as part of PLAN_REALTIRE Stream A.

use crate::terrain::{FlatTerrain, Terrain};

use super::super::internal::{
    add, cross, dot, finite_nonnegative, finite_or, finite_positive_or, finite_vec3, normalize,
    scale, sub,
};
use super::super::surface_material::sanitized_surface_material;
use super::super::{
    RealTireContact, RealTireTelemetrySample, RealWheelState, SidewallDihedral, StructuralEdge,
    SurfaceMaterial, TetraBeadSeatDiagnostics, TetraFemMatrixFreeNewtonStepReport,
    TetraSidewallDihedralDiagnostics, TetraSidewallLaminateAudit,
    TetraSidewallLaminateSanityReport, TetraSidewallLaminateStretchDiagnostics, TetraTireFem,
    TetraTireFemDiagnostics, TetraTireFemGpuSnapshot, TetraTireFemParams, TetraTireFemSolverMode,
    TireContactCell, TireSolverBudgetTelemetry, add_holzapfel_fiber_stress, brush_bristle_profile,
    brush_load_sensitivity, brush_pressure_patch_gain, brush_pressure_stiffness_factor, det3, inv3,
    lame_from_young_poisson, make_tetra_element, mat_vec, matrix_free_newton_correction,
    patch_flatten_nodes, patch_newton_residual_with_masses, patch_total_energy_with_masses,
    patch_unflatten_nodes, project_edge_length_constraints, project_sidewall_dihedral_constraints,
    project_sidewall_ply_strain_constraints, project_to_ground, quad_dihedral_angle,
    smooth_combined_brush_force, surface_material_with_terrain_film, tetra_signed_volume,
    transpose3, vec_max_abs, vec_norm,
};

impl TetraTireFem {
    pub fn new(hub_x: f64, pressure_pa: f64, mut params: TetraTireFemParams) -> Self {
        params.pressure_pa = pressure_pa.max(0.0);
        params.cavity_carcass_coupling =
            finite_or(params.cavity_carcass_coupling, 0.18).clamp(0.0, 1.0);
        params.segments = params.segments.max(8);
        params.radial_layers = params.radial_layers.max(2);
        params.width_layers = params.width_layers.max(2);
        params.inner_radius = params.inner_radius.clamp(0.02, params.outer_radius * 0.92);
        params.poisson_ratio = params.poisson_ratio.clamp(-0.9, 0.49);

        let n = params.segments;
        let nr = params.radial_layers;
        let nw = params.width_layers;
        let hub_y = params.outer_radius * 1.05;
        let hub = [hub_x, hub_y, 0.0];
        let node_count = n * nr * nw;
        let mut rest = Vec::with_capacity(node_count);
        let mut pos = Vec::with_capacity(node_count);
        let mut vel = Vec::with_capacity(node_count);
        let idx = |i: usize, r: usize, w: usize| -> usize { ((i % n) * nr + r) * nw + w };

        for i in 0..n {
            let theta = i as f64 / n as f64 * std::f64::consts::TAU;
            let dir = [theta.cos(), theta.sin()];
            for r in 0..nr {
                let q = if nr > 1 {
                    r as f64 / (nr - 1) as f64
                } else {
                    0.0
                };
                let radius = params.inner_radius + (params.outer_radius - params.inner_radius) * q;
                for w in 0..nw {
                    let wz = if nw > 1 {
                        w as f64 / (nw - 1) as f64
                    } else {
                        0.5
                    };
                    let p = [
                        hub_x + radius * dir[0],
                        hub_y + radius * dir[1],
                        -0.5 * params.width + params.width * wz,
                    ];
                    rest.push(p);
                    pos.push(p);
                    vel.push([0.0; 3]);
                }
            }
        }

        let mut tets = Vec::new();
        let mut mass = vec![0.0; node_count];
        // Stage (c) — anisotropic ply/belt parameters.
        //
        //   k₁  (Pa)         linear fibre stiffness when slack/just-taut
        //   k₂  (dim-less)   Holzapfel exponential — pretensioned fibres
        //                    progressively stiffen with further stretch
        //
        // Tread/belt and bead get strong circumferential fibres so cavity
        // hoop tension stays in the belt rather than ballooning the bead.
        // Tread belts are explicit ±20° crossed belt plies relative to the
        // circumferential direction, plus a circumferential cap/hoop ply and
        // a lighter axial stabilizer. Sidewalls are explicit ±38° bias plies
        // mixing radial and circumferential directions, plus radial and axial
        // helper plies. Pure-rubber carcass regions remain isotropic.
        const FIBER_K1_TREAD: f64 = 8.0e6;
        const FIBER_K2_TREAD: f64 = 25.0;
        const FIBER_K1_BEAD: f64 = 4.0e7;
        const FIBER_K2_BEAD: f64 = 15.0;
        const FIBER_K1_SIDEWALL: f64 = 2.0e6;
        const FIBER_K2_SIDEWALL: f64 = 12.0;
        let belt_angle = 20.0_f64.to_radians();
        let sidewall_angle = 38.0_f64.to_radians();
        for i in 0..n {
            let ni = (i + 1) % n;
            // Cell centroid angle — fibre direction is set from it once
            // per cell so all five tetrahedra in the cell share the
            // same belt direction.
            let theta_centroid = (i as f64 + 0.5) / n as f64 * std::f64::consts::TAU;
            let circ = [-theta_centroid.sin(), theta_centroid.cos(), 0.0];
            let radial = [theta_centroid.cos(), theta_centroid.sin(), 0.0];
            let axial = [0.0, 0.0, 1.0];
            let belt_a = normalize(add(
                scale(circ, belt_angle.cos()),
                scale(axial, belt_angle.sin()),
            ));
            let belt_b = normalize(add(
                scale(circ, belt_angle.cos()),
                scale(axial, -belt_angle.sin()),
            ));
            let sidewall_a = normalize(add(
                scale(radial, sidewall_angle.cos()),
                scale(circ, sidewall_angle.sin()),
            ));
            let sidewall_b = normalize(add(
                scale(radial, sidewall_angle.cos()),
                scale(circ, -sidewall_angle.sin()),
            ));
            for r in 0..(nr - 1) {
                for w in 0..(nw - 1) {
                    let (
                        young,
                        fiber_dir,
                        fiber_k1,
                        fiber_k2,
                        fiber_dir_b,
                        fiber_k1_b,
                        fiber_k2_b,
                        fiber_dir_c,
                        fiber_k1_c,
                        fiber_k2_c,
                        fiber_dir_d,
                        fiber_k1_d,
                        fiber_k2_d,
                    ) = if r == 0 {
                        (
                            params.bead_young_modulus,
                            circ,
                            FIBER_K1_BEAD * 0.5,
                            FIBER_K2_BEAD,
                            circ,
                            FIBER_K1_BEAD * 0.5,
                            FIBER_K2_BEAD,
                            axial,
                            FIBER_K1_BEAD * 0.08,
                            FIBER_K2_BEAD,
                            radial,
                            FIBER_K1_BEAD * 0.08,
                            FIBER_K2_BEAD,
                        )
                    } else if r + 1 >= nr - 1 {
                        (
                            params.tread_young_modulus,
                            belt_a,
                            FIBER_K1_TREAD * 0.55,
                            FIBER_K2_TREAD,
                            belt_b,
                            FIBER_K1_TREAD * 0.55,
                            FIBER_K2_TREAD,
                            circ,
                            FIBER_K1_TREAD * 0.35,
                            FIBER_K2_TREAD,
                            axial,
                            FIBER_K1_TREAD * 0.18,
                            FIBER_K2_TREAD,
                        )
                    } else if w == 0 || w + 1 >= nw - 1 {
                        (
                            params.sidewall_young_modulus,
                            sidewall_a,
                            FIBER_K1_SIDEWALL * 0.60,
                            FIBER_K2_SIDEWALL,
                            sidewall_b,
                            FIBER_K1_SIDEWALL * 0.60,
                            FIBER_K2_SIDEWALL,
                            radial,
                            FIBER_K1_SIDEWALL * 0.45,
                            FIBER_K2_SIDEWALL,
                            axial,
                            FIBER_K1_SIDEWALL * 0.20,
                            FIBER_K2_SIDEWALL,
                        )
                    } else {
                        (
                            params.young_modulus,
                            [0.0; 3],
                            0.0,
                            0.0,
                            [0.0; 3],
                            0.0,
                            0.0,
                            [0.0; 3],
                            0.0,
                            0.0,
                            [0.0; 3],
                            0.0,
                            0.0,
                        )
                    };
                    let (mu, lambda) = lame_from_young_poisson(young, params.poisson_ratio);
                    let a = idx(i, r, w);
                    let b = idx(ni, r, w);
                    let c = idx(ni, r + 1, w);
                    let d = idx(i, r + 1, w);
                    let e = idx(i, r, w + 1);
                    let f = idx(ni, r, w + 1);
                    let g = idx(ni, r + 1, w + 1);
                    let h = idx(i, r + 1, w + 1);
                    let cell_tets = [
                        [a, b, d, e],
                        [b, c, d, g],
                        [b, e, f, g],
                        [d, e, g, h],
                        [b, d, e, g],
                    ];
                    for ids in cell_tets {
                        if let Some(tet) = make_tetra_element(
                            ids,
                            &rest,
                            mu,
                            lambda,
                            fiber_dir,
                            fiber_k1,
                            fiber_k2,
                            fiber_dir_b,
                            fiber_k1_b,
                            fiber_k2_b,
                            fiber_dir_c,
                            fiber_k1_c,
                            fiber_k2_c,
                            fiber_dir_d,
                            fiber_k1_d,
                            fiber_k2_d,
                        ) {
                            let nodal_mass = params.density * tet.volume / 4.0;
                            for &id in &tet.ids {
                                mass[id] += nodal_mass;
                            }
                            tets.push(tet);
                        }
                    }
                }
            }
        }
        for m in &mut mass {
            *m = m.max(0.02);
        }

        let mut surface_edges = Vec::new();
        for i in 0..n {
            let ni = (i + 1) % n;
            // Outer tread circumferential and width lines.
            for w in 0..nw {
                surface_edges.push((idx(i, nr - 1, w), idx(ni, nr - 1, w)));
            }
            for w in 0..(nw - 1) {
                surface_edges.push((idx(i, nr - 1, w), idx(i, nr - 1, w + 1)));
            }
            // Sidewall radial lines on both side faces.
            for r in 0..(nr - 1) {
                surface_edges.push((idx(i, r, 0), idx(i, r + 1, 0)));
                surface_edges.push((idx(i, r, nw - 1), idx(i, r + 1, nw - 1)));
            }
            // Inner bead/rim circumferential hints.
            surface_edges.push((idx(i, 0, 0), idx(ni, 0, 0)));
            surface_edges.push((idx(i, 0, nw - 1), idx(ni, 0, nw - 1)));
        }

        let mut tread_belt_edges = Vec::new();
        let mut push_belt_edge = |a: usize, b: usize| {
            let rest_length = dot(sub(rest[b], rest[a]), sub(rest[b], rest[a])).sqrt();
            if rest_length > 1.0e-9 {
                tread_belt_edges.push(StructuralEdge {
                    ids: [a, b],
                    rest_length,
                });
            }
        };
        for i in 0..n {
            let ni = (i + 1) % n;
            for w in 0..nw {
                push_belt_edge(idx(i, nr - 1, w), idx(ni, nr - 1, w));
            }
            for w in 0..(nw - 1) {
                push_belt_edge(idx(i, nr - 1, w), idx(i, nr - 1, w + 1));
                // Crossed belt diagonals resist tread-band shear and give
                // the XPBD projection an explicit ± belt-like structure.
                push_belt_edge(idx(i, nr - 1, w), idx(ni, nr - 1, w + 1));
                push_belt_edge(idx(ni, nr - 1, w), idx(i, nr - 1, w + 1));
            }
        }

        let mut sidewall_bend_edges = Vec::new();
        let mut push_sidewall_bend_edge = |a: usize, b: usize| {
            let rest_length = dot(sub(rest[b], rest[a]), sub(rest[b], rest[a])).sqrt();
            if rest_length > 1.0e-9 {
                sidewall_bend_edges.push(StructuralEdge {
                    ids: [a, b],
                    rest_length,
                });
            }
        };
        let mut sidewall_dihedrals = Vec::new();
        let mut push_sidewall_dihedral = |a: usize, b: usize, c: usize, d: usize| {
            sidewall_dihedrals.push(SidewallDihedral {
                ids: [a, b, c, d],
                rest_angle_rad: quad_dihedral_angle(&rest, [a, b, c, d]),
            });
        };
        for i in 0..n {
            let ni = (i + 1) % n;
            for &w in &[0, nw - 1] {
                for r in 0..(nr - 1) {
                    push_sidewall_dihedral(
                        idx(i, r, w),
                        idx(ni, r, w),
                        idx(ni, r + 1, w),
                        idx(i, r + 1, w),
                    );
                    // Opposite vertices of a sidewall quad. Preserving this
                    // rest chord is a cheap XPBD hinge/bending proxy: it
                    // resists sidewall folding without pretending to be the
                    // final dihedral-angle or laminated sidewall solver.
                    push_sidewall_bend_edge(idx(ni, r, w), idx(i, r + 1, w));
                    push_sidewall_bend_edge(idx(i, r, w), idx(ni, r + 1, w));
                }
            }
        }

        let mut tread_nodes = Vec::with_capacity(n * nw);
        for i in 0..n {
            for w in 0..nw {
                tread_nodes.push(idx(i, nr - 1, w));
            }
        }
        let mut bead_offsets = Vec::with_capacity(n * nw);
        for i in 0..n {
            for w in 0..nw {
                let id = idx(i, 0, w);
                bead_offsets.push((id, sub(rest[id], hub)));
            }
        }

        // ── Cavity surface triangulation ────────────────────────────────
        // The gas pocket lives between the rim and the bead-layer (`r=0`).
        // We close it with two virtual axial caps anchored at the hub axis
        // (the rim is rigid and rides with the hub). Wind every triangle
        // so its normal points *out* of the cavity → into the rubber. With
        // that convention `(b-a) × (c-a)` is the pressure-load direction
        // applied at each vertex in `step_inner`.
        let mut cavity_tube_tris = Vec::with_capacity(n * (nw - 1) * 2);
        for i in 0..n {
            let ni = (i + 1) % n;
            for w in 0..(nw - 1) {
                let a = idx(i, 0, w) as u32;
                let b = idx(ni, 0, w) as u32;
                let c = idx(ni, 0, w + 1) as u32;
                let d = idx(i, 0, w + 1) as u32;
                cavity_tube_tris.push([a, b, c]);
                cavity_tube_tris.push([a, c, d]);
            }
        }
        let mut cavity_neg_edges = Vec::with_capacity(n);
        let mut cavity_pos_edges = Vec::with_capacity(n);
        for i in 0..n {
            let ni = (i + 1) % n;
            cavity_neg_edges.push([idx(i, 0, 0) as u32, idx(ni, 0, 0) as u32]);
            cavity_pos_edges.push([idx(i, 0, nw - 1) as u32, idx(ni, 0, nw - 1) as u32]);
        }

        // Stage (d) — tread-node bookkeeping.
        let mut node_to_tread: Vec<Option<u32>> = vec![None; node_count];
        for (ti, &id) in tread_nodes.iter().enumerate() {
            node_to_tread[id] = Some(ti as u32);
        }
        let tread_bristle = vec![[0.0_f64; 3]; tread_nodes.len()];
        // Tread surface area / count → area per tread node.
        let tread_area_total = std::f64::consts::TAU * params.outer_radius * params.width;
        let tread_area_per_node = if !tread_nodes.is_empty() {
            tread_area_total / tread_nodes.len() as f64
        } else {
            0.0
        };

        // PLAN_ROLL Stage A — body-frame state defaults to the rest
        // pose, expressed as offsets from the hub so the rotation
        // transform `world = R(hub_angle) · body + hub` works
        // naturally. With `hub_angle = 0` body_pos[i] = rest[i] − hub,
        // and world_pos[i] = body_pos[i] + hub = rest[i]. Stored
        // before `rest` is moved into the struct below.
        let body_pos_init: Vec<[f64; 3]> = rest
            .iter()
            .map(|p| [p[0] - hub[0], p[1] - hub[1], p[2] - hub[2]])
            .collect();
        let body_vel_init = vec![[0.0_f64; 3]; rest.len()];
        let mut tire = Self {
            params,
            hub,
            hub_v_y: 0.0,
            hub_v_xz: [0.0; 2],
            // PLAN_TOURQUE Stage A — torque/spin state starts at rest.
            hub_angle: 0.0,
            hub_omega: 0.0,
            drive_torque_n_m: 0.0,
            brake_torque_n_m: 0.0,
            hub_reaction_xyz: [0.0; 3],
            // PLAN_TOURQUE Stage D — kinematic-by-default. Scenes that
            // want the hub to roll from contact reaction call
            // `set_hub_dynamic()`.
            hub_kinematic: true,
            external_force_xyz: [0.0; 3],
            rest,
            pos,
            vel,
            // PLAN_ROLL Stage A — body-frame state defaults to the
            // rest pose with zero velocity. Stages B+ wire these into
            // step_inner.
            body_pos: body_pos_init,
            body_vel: body_vel_init,
            mass,
            tets,
            surface_edges,
            tread_belt_edges,
            sidewall_bend_edges,
            sidewall_dihedrals,
            tread_nodes,
            bead_offsets,
            cavity_tube_tris,
            cavity_neg_edges,
            cavity_pos_edges,
            cavity_volume_rest: 0.0,
            tread_bristle,
            node_to_tread,
            tread_area_per_node,
            last_xpbd_volume_correction_m: 0.0,
            last_xpbd_bead_correction_m: 0.0,
            last_xpbd_structural_correction_m: 0.0,
            scratch_force: Vec::new(),
            scratch_node_stiffness_diag: Vec::new(),
        };
        tire.cavity_volume_rest = tire.cavity_volume();
        tire
    }

    /// Read-only view of the surface material driving brush contact.
    pub fn surface_material(&self) -> SurfaceMaterial {
        self.params.surface_material
    }

    /// Replace the surface material at runtime — e.g. drive onto ice.
    pub fn set_surface_material(&mut self, material: SurfaceMaterial) {
        self.params.surface_material = material;
    }

    /// Stage (b) leftover — the isotropic stiffening proxy that the GPU
    /// shader adds to each tet's μ. Lives in a uniform on GPU so the
    /// persistent buffers (stage (g)) don't have to re-upload `tet_mu`
    /// every time the pressure changes.
    pub fn belt_proxy_mu(&self) -> f64 {
        0.25 * self.params.pressure_pa.max(0.0)
    }

    /// Enclosed cavity volume (m³) — divergence-theorem fan from the hub
    /// over the bead tube and the two virtual axial end-caps. Used by
    /// the gas-law pressure traction in `step_inner`.
    pub fn cavity_volume(&self) -> f64 {
        let half_w = self.params.width * 0.5;
        let neg_axis = [self.hub[0], self.hub[1], -half_w];
        let pos_axis = [self.hub[0], self.hub[1], half_w];
        let rel = |p: [f64; 3]| sub(p, self.hub);
        let mut v = 0.0;
        for tri in &self.cavity_tube_tris {
            let a = rel(self.pos[tri[0] as usize]);
            let b = rel(self.pos[tri[1] as usize]);
            let c = rel(self.pos[tri[2] as usize]);
            v += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]))
                / 6.0;
        }
        let neg_rel = rel(neg_axis);
        let pos_rel = rel(pos_axis);
        for edge in &self.cavity_neg_edges {
            let a = neg_rel;
            let b = rel(self.pos[edge[1] as usize]);
            let c = rel(self.pos[edge[0] as usize]);
            v += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]))
                / 6.0;
        }
        for edge in &self.cavity_pos_edges {
            let a = pos_rel;
            let b = rel(self.pos[edge[0] as usize]);
            let c = rel(self.pos[edge[1] as usize]);
            v += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]))
                / 6.0;
        }
        v.abs().max(1.0e-9)
    }

    fn effective_cavity_pressure_pa(&self) -> f64 {
        let p_ref = self.params.pressure_pa.max(0.0);
        if p_ref <= 0.0 || self.cavity_volume_rest <= 0.0 {
            return 0.0;
        }
        let v_now = self.cavity_volume().max(1.0e-12);
        let gamma = self.params.cavity_polytropic_gamma.max(0.1);
        let ratio = (self.cavity_volume_rest / v_now).powf(gamma);
        let coupling = self.params.cavity_carcass_coupling.clamp(0.0, 1.0);
        (p_ref * ratio * coupling).clamp(0.0, 50.0 * p_ref.max(1.0e4))
    }

    fn cavity_radial_force_per_bead_n(&self) -> f64 {
        if self.bead_offsets.is_empty() {
            return 0.0;
        }
        let bead_surface_area =
            std::f64::consts::TAU * self.params.inner_radius * self.params.width;
        let area_per_bead = bead_surface_area / self.bead_offsets.len() as f64;
        self.effective_cavity_pressure_pa() * area_per_bead
    }

    fn max_tet_volume_error_m3(&self) -> f64 {
        let mut max_error: f64 = 0.0;
        for tet in &self.tets {
            let volume = tetra_signed_volume(tet.ids, &self.pos);
            if volume.is_finite() {
                max_error = max_error.max((volume - tet.volume).abs());
            }
        }
        max_error
    }

    pub fn bead_mount_error_stats_m(&self) -> (f64, f64) {
        if self.bead_offsets.is_empty() {
            return (0.0, 0.0);
        }
        let mut sum_error = 0.0;
        let mut max_error = 0.0_f64;
        for &(id, offset) in &self.bead_offsets {
            let target = add(self.hub, offset);
            let delta = sub(self.pos[id], target);
            let error = dot(delta, delta).sqrt();
            if error.is_finite() {
                sum_error += error;
                max_error = max_error.max(error);
            }
        }
        (sum_error / self.bead_offsets.len() as f64, max_error)
    }

    pub fn bead_seat_diagnostics(&self) -> TetraBeadSeatDiagnostics {
        if self.bead_offsets.is_empty() {
            return TetraBeadSeatDiagnostics::default();
        }
        let mut sum_radial = 0.0_f64;
        let mut max_radial = 0.0_f64;
        let mut sum_axial = 0.0_f64;
        let mut max_axial = 0.0_f64;
        let mut max_tangential = 0.0_f64;
        let axial = [0.0, 0.0, 1.0];
        let mut count = 0usize;
        for &(id, offset) in &self.bead_offsets {
            let target = add(self.hub, offset);
            let delta = sub(self.pos[id], target);
            if !finite_vec3(delta) {
                continue;
            }
            let radial = normalize([offset[0], offset[1], 0.0]);
            let tangent = normalize(cross(axial, radial));
            let radial_error = dot(delta, radial).abs();
            let axial_error = dot(delta, axial).abs();
            let tangential_error = dot(delta, tangent).abs();
            if radial_error.is_finite() && axial_error.is_finite() && tangential_error.is_finite() {
                sum_radial += radial_error;
                max_radial = max_radial.max(radial_error);
                sum_axial += axial_error;
                max_axial = max_axial.max(axial_error);
                max_tangential = max_tangential.max(tangential_error);
                count += 1;
            }
        }
        if count == 0 {
            return TetraBeadSeatDiagnostics {
                bead_node_count: self.bead_offsets.len(),
                ..Default::default()
            };
        }
        TetraBeadSeatDiagnostics {
            bead_node_count: count,
            mean_abs_radial_error_m: sum_radial / count as f64,
            max_abs_radial_error_m: max_radial,
            mean_abs_axial_error_m: sum_axial / count as f64,
            max_abs_axial_error_m: max_axial,
            max_tangential_error_m: max_tangential,
        }
    }

    pub fn diagnostics(&self) -> TetraTireFemDiagnostics {
        let cavity_volume_m3 = self.cavity_volume();
        let (mean_bead_mount_error_m, max_bead_mount_error_m) = self.bead_mount_error_stats_m();
        TetraTireFemDiagnostics {
            pressure_pa: self.params.pressure_pa,
            cavity_volume_m3,
            cavity_rest_volume_m3: self.cavity_volume_rest,
            cavity_volume_ratio: if self.cavity_volume_rest > 1.0e-12 {
                cavity_volume_m3 / self.cavity_volume_rest
            } else {
                0.0
            },
            effective_cavity_pressure_pa: self.effective_cavity_pressure_pa(),
            cavity_carcass_coupling: self.params.cavity_carcass_coupling,
            cavity_radial_force_per_bead_n: self.cavity_radial_force_per_bead_n(),
            max_tet_volume_error_m3: self.max_tet_volume_error_m3(),
            mean_bead_mount_error_m,
            max_bead_mount_error_m,
            last_xpbd_volume_correction_m: self.last_xpbd_volume_correction_m,
            last_xpbd_bead_correction_m: self.last_xpbd_bead_correction_m,
            last_xpbd_structural_correction_m: self.last_xpbd_structural_correction_m,
        }
    }

    pub fn max_node_speed_m_per_s(&self) -> f64 {
        self.vel
            .iter()
            .map(|v| dot(*v, *v).sqrt())
            .filter(|v| v.is_finite())
            .fold(0.0, f64::max)
    }

    pub fn solver_budget_telemetry(
        &self,
        substeps_per_frame: usize,
        contact_telemetry: &RealTireTelemetrySample,
    ) -> TireSolverBudgetTelemetry {
        TireSolverBudgetTelemetry::from_tetra_fem(
            &self.params,
            &self.diagnostics(),
            self.max_node_speed_m_per_s(),
            substeps_per_frame,
            contact_telemetry,
        )
    }

    pub fn sidewall_laminate_audit(&self) -> TetraSidewallLaminateAudit {
        let target_bias_rad = 38.0_f64.to_radians();
        let mut count = 0usize;
        let mut crossed = 0usize;
        let mut sum_bias = 0.0;
        let mut max_bias_error = 0.0_f64;
        let mut sum_radial_align = 0.0;
        let mut sum_axial_align = 0.0;
        for tet in &self.tets {
            if tet.fiber_k1 <= 0.0
                || tet.fiber_k1_b <= 0.0
                || tet.fiber_k1_c <= 0.0
                || tet.fiber_k1_d <= 0.0
            {
                continue;
            }
            let centroid = tet
                .ids
                .iter()
                .fold([0.0; 3], |acc, &id| add(acc, scale(self.rest[id], 0.25)));
            let radial_raw = [centroid[0] - self.hub[0], centroid[1] - self.hub[1], 0.0];
            let radial_len = dot(radial_raw, radial_raw).sqrt();
            if !radial_len.is_finite() || radial_len <= 1.0e-9 {
                continue;
            }
            let radial = scale(radial_raw, 1.0 / radial_len);
            let circ = [-radial[1], radial[0], 0.0];
            let axial = [0.0, 0.0, 1.0];
            let radial_helper = dot(tet.fiber_dir_c, radial).abs();
            let axial_helper = dot(tet.fiber_dir_d, axial).abs();
            // Tread/belt elements use circumferential + axial helpers; bead
            // elements use hoop helpers. Sidewall laminate elements are the
            // ones with radial + axial helpers.
            if radial_helper < 0.85 || axial_helper < 0.85 {
                continue;
            }
            let bias_a = dot(tet.fiber_dir, radial).abs().clamp(-1.0, 1.0).acos();
            let bias_b = dot(tet.fiber_dir_b, radial).abs().clamp(-1.0, 1.0).acos();
            let mean_bias = 0.5 * (bias_a + bias_b);
            let sign_a = dot(tet.fiber_dir, circ).signum();
            let sign_b = dot(tet.fiber_dir_b, circ).signum();
            if sign_a != 0.0 && sign_b != 0.0 && sign_a != sign_b {
                crossed += 1;
            }
            count += 1;
            sum_bias += mean_bias;
            max_bias_error = max_bias_error.max((bias_a - target_bias_rad).abs());
            max_bias_error = max_bias_error.max((bias_b - target_bias_rad).abs());
            sum_radial_align += radial_helper;
            sum_axial_align += axial_helper;
        }
        if count == 0 {
            return TetraSidewallLaminateAudit::default();
        }
        let inv = 1.0 / count as f64;
        TetraSidewallLaminateAudit {
            sidewall_element_count: count,
            crossed_pair_count: crossed,
            mean_bias_angle_deg: (sum_bias * inv).to_degrees(),
            max_bias_angle_error_deg: max_bias_error.to_degrees(),
            mean_radial_helper_alignment: sum_radial_align * inv,
            mean_axial_helper_alignment: sum_axial_align * inv,
        }
    }

    pub fn sidewall_laminate_stretch_diagnostics(&self) -> TetraSidewallLaminateStretchDiagnostics {
        let mut sidewall_count = 0usize;
        let mut tensile_count = 0usize;
        let mut tensile_sum = 0.0_f64;
        let mut max_tensile = 0.0_f64;
        let mut max_compressive = 0.0_f64;
        for tet in &self.tets {
            if tet.fiber_k1 <= 0.0
                || tet.fiber_k1_b <= 0.0
                || tet.fiber_k1_c <= 0.0
                || tet.fiber_k1_d <= 0.0
            {
                continue;
            }
            let centroid = tet
                .ids
                .iter()
                .fold([0.0; 3], |acc, &id| add(acc, scale(self.rest[id], 0.25)));
            let radial_raw = [centroid[0] - self.hub[0], centroid[1] - self.hub[1], 0.0];
            let radial_len = dot(radial_raw, radial_raw).sqrt();
            if !radial_len.is_finite() || radial_len <= 1.0e-9 {
                continue;
            }
            let radial = scale(radial_raw, 1.0 / radial_len);
            let axial = [0.0, 0.0, 1.0];
            let radial_helper = dot(tet.fiber_dir_c, radial).abs();
            let axial_helper = dot(tet.fiber_dir_d, axial).abs();
            if radial_helper < 0.85 || axial_helper < 0.85 {
                continue;
            }
            let mut f_def = [[0.0; 3]; 3];
            for a in 0..4 {
                let id = tet.ids[a];
                for row in 0..3 {
                    for col in 0..3 {
                        f_def[row][col] += self.pos[id][row] * tet.grads[a][col];
                    }
                }
            }
            if det3(f_def).abs() <= 1.0e-12 {
                continue;
            }
            sidewall_count += 1;
            for dir in [
                tet.fiber_dir,
                tet.fiber_dir_b,
                tet.fiber_dir_c,
                tet.fiber_dir_d,
            ] {
                let fa = mat_vec(f_def, dir);
                let strain = dot(fa, fa).sqrt() - 1.0;
                if !strain.is_finite() {
                    continue;
                }
                if strain > 0.0 {
                    tensile_sum += strain;
                    tensile_count += 1;
                    max_tensile = max_tensile.max(strain);
                } else {
                    max_compressive = max_compressive.max(-strain);
                }
            }
        }
        TetraSidewallLaminateStretchDiagnostics {
            sidewall_element_count: sidewall_count,
            tensile_family_count: tensile_count,
            mean_tensile_strain: if tensile_count > 0 {
                tensile_sum / tensile_count as f64
            } else {
                0.0
            },
            max_tensile_strain: max_tensile,
            max_compressive_strain: max_compressive,
        }
    }

    pub fn sidewall_laminate_sanity_report(&self) -> TetraSidewallLaminateSanityReport {
        let audit = self.sidewall_laminate_audit();
        let dihedral = self.sidewall_dihedral_diagnostics();
        let stretch = self.sidewall_laminate_stretch_diagnostics();
        let crossed_coverage = if audit.sidewall_element_count > 0 {
            audit.crossed_pair_count as f64 / audit.sidewall_element_count as f64
        } else {
            0.0
        };
        let helper_alignment_min = audit
            .mean_radial_helper_alignment
            .min(audit.mean_axial_helper_alignment);
        let fold_mean_limit_rad = 0.08;
        let fold_max_limit_rad = 0.25;
        let stretch_limit = 0.75;
        let passed = audit.is_finite()
            && dihedral.is_finite()
            && stretch.is_finite()
            && audit.sidewall_element_count > self.segments()
            && crossed_coverage > 0.98
            && (audit.mean_bias_angle_deg - 38.0).abs() < 1.0
            && audit.max_bias_angle_error_deg < 6.0
            && helper_alignment_min > 0.98
            && dihedral.mean_abs_deviation_rad <= fold_mean_limit_rad
            && dihedral.max_abs_deviation_rad <= fold_max_limit_rad
            && stretch.max_tensile_strain <= stretch_limit
            && stretch.max_compressive_strain <= stretch_limit;
        TetraSidewallLaminateSanityReport {
            audit,
            dihedral,
            stretch,
            crossed_coverage,
            helper_alignment_min,
            fold_mean_limit_rad,
            fold_max_limit_rad,
            stretch_limit,
            passed,
        }
    }

    pub fn set_pressure(&mut self, pressure_pa: f64) {
        self.params.pressure_pa = pressure_pa.max(0.0);
    }

    pub fn pressure(&self) -> f64 {
        self.params.pressure_pa
    }

    pub fn semi_implicit_stiffness_scale(&self) -> f64 {
        self.params.semi_implicit_stiffness_scale
    }

    pub fn hub(&self) -> [f64; 3] {
        self.hub
    }

    /// Current hub translational velocity in world XYZ (m/s). The hub
    /// integrates X/Z from the contact-patch brush reaction and Y from
    /// downforce + gravity + bead reactions when the hub is dynamic.
    pub fn hub_velocity(&self) -> [f64; 3] {
        [self.hub_v_xz[0], self.hub_v_y, self.hub_v_xz[1]]
    }

    pub fn nodes(&self) -> &[[f64; 3]] {
        &self.pos
    }

    /// PLAN_ROLL Stage A — body-frame node positions. With `hub_angle`
    /// at zero, body == world − hub. Stages B+ make this diverge from
    /// `nodes()` once the rotation transform comes online.
    pub fn body_nodes(&self) -> &[[f64; 3]] {
        &self.body_pos
    }

    /// PLAN_ROLL Stage A — body-frame node velocities. Companion to
    /// `body_nodes`. The world-frame velocity is
    /// `R(hub_angle) · body_v + ω × (R · body_pos) + hub_velocity`
    /// (Stage B implements that transform).
    pub fn body_velocities(&self) -> &[[f64; 3]] {
        &self.body_vel
    }

    /// PLAN_ROLL Stage B — transform a body-frame point into the world
    /// frame using the current `hub_angle` rotation and `hub`
    /// translation. Sign convention: positive `hub_angle` ⇔ forward
    /// rolling (top of wheel +X, bottom −X), matching the CPU
    /// contact-slip code and the scene 123 spoke gizmo. The rotation
    /// is around +Z (spin axis); Z passes through unchanged.
    pub fn body_to_world(&self, body: [f64; 3]) -> [f64; 3] {
        let (s, c) = self.hub_angle.sin_cos();
        [
            c * body[0] + s * body[1] + self.hub[0],
            -s * body[0] + c * body[1] + self.hub[1],
            body[2] + self.hub[2],
        ]
    }

    /// PLAN_ROLL Stage B — inverse of `body_to_world`. Transform a
    /// world-frame point back into the body frame.
    pub fn world_to_body(&self, world: [f64; 3]) -> [f64; 3] {
        let (s, c) = self.hub_angle.sin_cos();
        let lx = world[0] - self.hub[0];
        let ly = world[1] - self.hub[1];
        [c * lx - s * ly, s * lx + c * ly, world[2] - self.hub[2]]
    }

    /// PLAN_ROLL Stage B — transform a body-frame velocity into the
    /// world frame, including the rotation-induced
    /// `ω × (R · body_pos)` term and the hub translational velocity.
    /// `body_pos` is the body-frame position the velocity is measured
    /// at (needed because the spin contribution depends on radius).
    pub fn body_velocity_to_world(&self, body_pos: [f64; 3], body_vel: [f64; 3]) -> [f64; 3] {
        let (s, c) = self.hub_angle.sin_cos();
        // Rotated body position (X/Y only — spin is around +Z).
        let rx = c * body_pos[0] + s * body_pos[1];
        let ry = -s * body_pos[0] + c * body_pos[1];
        // ω × r for ω = hub_omega · (+Z spin) under the sign convention
        // pinned in `integrate_spin_dof`: ω×r at world-frame radius r
        // is `[ω · r_y, −ω · r_x, 0]`.
        let spin_v_x = self.hub_omega * ry;
        let spin_v_y = -self.hub_omega * rx;
        // Rotated body velocity (X/Y).
        let rvx = c * body_vel[0] + s * body_vel[1];
        let rvy = -s * body_vel[0] + c * body_vel[1];
        let hub_v = [self.hub_v_xz[0], self.hub_v_y, self.hub_v_xz[1]];
        [
            rvx + spin_v_x + hub_v[0],
            rvy + spin_v_y + hub_v[1],
            body_vel[2] + hub_v[2],
        ]
    }

    pub fn surface_edges(&self) -> &[(usize, usize)] {
        &self.surface_edges
    }

    pub fn tread_nodes(&self) -> &[usize] {
        &self.tread_nodes
    }

    pub fn tetra_count(&self) -> usize {
        self.tets.len()
    }

    pub fn segments(&self) -> usize {
        self.params.segments
    }

    pub fn width(&self) -> f64 {
        self.params.width
    }

    pub fn outer_radius(&self) -> f64 {
        self.params.outer_radius
    }

    pub fn sidewall_dihedral_diagnostics(&self) -> TetraSidewallDihedralDiagnostics {
        if self.sidewall_dihedrals.is_empty() {
            return TetraSidewallDihedralDiagnostics::default();
        }
        let mut sum = 0.0_f64;
        let mut max_dev = 0.0_f64;
        for diag in &self.sidewall_dihedrals {
            let dev = (quad_dihedral_angle(&self.pos, diag.ids) - diag.rest_angle_rad).abs();
            if dev.is_finite() {
                sum += dev;
                max_dev = max_dev.max(dev);
            }
        }
        TetraSidewallDihedralDiagnostics {
            quad_count: self.sidewall_dihedrals.len(),
            mean_abs_deviation_rad: sum / self.sidewall_dihedrals.len() as f64,
            max_abs_deviation_rad: max_dev,
        }
    }

    pub fn gpu_snapshot(&self) -> TetraTireFemGpuSnapshot {
        let vec4 = |p: [f64; 3]| -> [f32; 4] { [p[0] as f32, p[1] as f32, p[2] as f32, 0.0] };
        let mut tet_indices = Vec::with_capacity(self.tets.len() * 4);
        let mut tet_grads = Vec::with_capacity(self.tets.len() * 4);
        let mut tet_volumes = Vec::with_capacity(self.tets.len());
        let mut tet_mu = Vec::with_capacity(self.tets.len());
        let mut tet_lambda = Vec::with_capacity(self.tets.len());
        let mut tet_fiber_dir = Vec::with_capacity(self.tets.len());
        let mut tet_fiber_k1 = Vec::with_capacity(self.tets.len());
        let mut tet_fiber_k2 = Vec::with_capacity(self.tets.len());
        let mut tet_fiber_dir_b = Vec::with_capacity(self.tets.len());
        let mut tet_fiber_k1_b = Vec::with_capacity(self.tets.len());
        let mut tet_fiber_k2_b = Vec::with_capacity(self.tets.len());
        let mut tet_fiber_dir_c = Vec::with_capacity(self.tets.len());
        let mut tet_fiber_k1_c = Vec::with_capacity(self.tets.len());
        let mut tet_fiber_k2_c = Vec::with_capacity(self.tets.len());
        let mut tet_fiber_dir_d = Vec::with_capacity(self.tets.len());
        let mut tet_fiber_k1_d = Vec::with_capacity(self.tets.len());
        let mut tet_fiber_k2_d = Vec::with_capacity(self.tets.len());
        // `tet_mu` is the *pure* per-element rubber μ — the pressure-
        // dependent belt-proxy stiffening is carried in `belt_proxy_mu`
        // (a uniform), so the persistent GPU path does not have to re-
        // upload `tet_mu` when the inflation pressure changes.
        for tet in &self.tets {
            for &id in &tet.ids {
                tet_indices.push(id as u32);
            }
            for &g in &tet.grads {
                tet_grads.push(vec4(g));
            }
            tet_volumes.push(tet.volume as f32);
            tet_mu.push(tet.mu as f32);
            tet_lambda.push(tet.lambda as f32);
            tet_fiber_dir.push(vec4(tet.fiber_dir));
            tet_fiber_k1.push(tet.fiber_k1 as f32);
            tet_fiber_k2.push(tet.fiber_k2 as f32);
            tet_fiber_dir_b.push(vec4(tet.fiber_dir_b));
            tet_fiber_k1_b.push(tet.fiber_k1_b as f32);
            tet_fiber_k2_b.push(tet.fiber_k2_b as f32);
            tet_fiber_dir_c.push(vec4(tet.fiber_dir_c));
            tet_fiber_k1_c.push(tet.fiber_k1_c as f32);
            tet_fiber_k2_c.push(tet.fiber_k2_c as f32);
            tet_fiber_dir_d.push(vec4(tet.fiber_dir_d));
            tet_fiber_k1_d.push(tet.fiber_k1_d as f32);
            tet_fiber_k2_d.push(tet.fiber_k2_d as f32);
        }

        // Stage (h) — build the node→tet CSR adjacency. Entry packs
        // `(tet_index << 2) | local_vertex_index ∈ {0,1,2,3}` so the
        // kernel can index `tet_indices` and `tet_grads` directly.
        let n_nodes = self.pos.len();
        let mut degree = vec![0_u32; n_nodes];
        for tet in &self.tets {
            for &id in &tet.ids {
                degree[id] += 1;
            }
        }
        let mut node_tet_offsets = Vec::with_capacity(n_nodes + 1);
        node_tet_offsets.push(0);
        let mut running: u32 = 0;
        for &d in &degree {
            running += d;
            node_tet_offsets.push(running);
        }
        let total: usize = running as usize;
        let mut node_tet_entries = vec![0_u32; total];
        let mut cursor = node_tet_offsets.clone();
        for (t_idx, tet) in self.tets.iter().enumerate() {
            for (local, &id) in tet.ids.iter().enumerate() {
                let pos_in_csr = &mut cursor[id];
                node_tet_entries[*pos_in_csr as usize] = ((t_idx as u32) << 2) | (local as u32);
                *pos_in_csr += 1;
            }
        }

        // Stage (f) — structural XPBD edge buffers for GPU belt/sidewall
        // parity. Per-edge compliance lets one kernel handle both the tread
        // belt and sidewall bend-proxy families.
        let mut xpbd_edge_indices =
            Vec::with_capacity(self.tread_belt_edges.len() + self.sidewall_bend_edges.len());
        let mut xpbd_edge_params =
            Vec::with_capacity(self.tread_belt_edges.len() + self.sidewall_bend_edges.len());
        for edge in &self.tread_belt_edges {
            xpbd_edge_indices.push([edge.ids[0] as u32, edge.ids[1] as u32, 0, 0]);
            xpbd_edge_params.push([
                edge.rest_length as f32,
                self.params.xpbd_tread_belt_compliance as f32,
                0.0,
                0.0,
            ]);
        }
        for edge in &self.sidewall_bend_edges {
            xpbd_edge_indices.push([edge.ids[0] as u32, edge.ids[1] as u32, 1, 0]);
            xpbd_edge_params.push([
                edge.rest_length as f32,
                self.params.xpbd_sidewall_bend_compliance as f32,
                0.0,
                0.0,
            ]);
        }
        let mut edge_degree = vec![0_u32; n_nodes];
        for edge in &xpbd_edge_indices {
            edge_degree[edge[0] as usize] += 1;
            edge_degree[edge[1] as usize] += 1;
        }
        let mut node_edge_offsets = Vec::with_capacity(n_nodes + 1);
        node_edge_offsets.push(0);
        let mut edge_running: u32 = 0;
        for &d in &edge_degree {
            edge_running += d;
            node_edge_offsets.push(edge_running);
        }
        let mut node_edge_entries = vec![0_u32; edge_running as usize];
        let mut edge_cursor = node_edge_offsets.clone();
        for (edge_idx, edge) in xpbd_edge_indices.iter().enumerate() {
            for local in 0..2 {
                let id = edge[local] as usize;
                let pos_in_csr = &mut edge_cursor[id];
                node_edge_entries[*pos_in_csr as usize] = ((edge_idx as u32) << 1) | local as u32;
                *pos_in_csr += 1;
            }
        }

        // Stage (f3) — sidewall dihedral-angle XPBD buffers for GPU
        // parity. This is a true quad fold-angle constraint, separate
        // from the older opposite-vertex edge-distance bending proxy.
        let mut xpbd_dihedral_indices = Vec::with_capacity(self.sidewall_dihedrals.len());
        let mut xpbd_dihedral_params = Vec::with_capacity(self.sidewall_dihedrals.len());
        for diag in &self.sidewall_dihedrals {
            xpbd_dihedral_indices.push([
                diag.ids[0] as u32,
                diag.ids[1] as u32,
                diag.ids[2] as u32,
                diag.ids[3] as u32,
            ]);
            xpbd_dihedral_params.push([
                diag.rest_angle_rad as f32,
                self.params.xpbd_sidewall_dihedral_compliance as f32,
                0.0,
                0.0,
            ]);
        }
        let mut dihedral_degree = vec![0_u32; n_nodes];
        for diag in &xpbd_dihedral_indices {
            for &id in diag {
                dihedral_degree[id as usize] += 1;
            }
        }
        let mut node_dihedral_offsets = Vec::with_capacity(n_nodes + 1);
        node_dihedral_offsets.push(0);
        let mut dihedral_running: u32 = 0;
        for &d in &dihedral_degree {
            dihedral_running += d;
            node_dihedral_offsets.push(dihedral_running);
        }
        let mut node_dihedral_entries = vec![0_u32; dihedral_running as usize];
        let mut dihedral_cursor = node_dihedral_offsets.clone();
        for (diag_idx, diag) in xpbd_dihedral_indices.iter().enumerate() {
            for local in 0..4 {
                let id = diag[local] as usize;
                let pos_in_csr = &mut dihedral_cursor[id];
                node_dihedral_entries[*pos_in_csr as usize] =
                    ((diag_idx as u32) << 2) | local as u32;
                *pos_in_csr += 1;
            }
        }
        let arc = self.params.outer_radius * std::f64::consts::TAU / self.params.segments as f64;
        let pressure_area_per_tread_node = arc * self.params.width
            / self.params.width_layers as f64
            * self.params.pressure_load_scale;

        // Stage (b) — pre-compute the cavity-pressure radial load per bead
        // node from the current cavity volume. Mirrors `step_inner`'s
        // formula, but distributed evenly across bead nodes so the GPU
        // shader only needs `(P_now·carcass_leak·A_per_bead)` as a scalar.
        let cavity_radial_force_per_bead = self.cavity_radial_force_per_bead_n() as f32;
        TetraTireFemGpuSnapshot {
            rest_positions: self.rest.iter().copied().map(vec4).collect(),
            positions: self.pos.iter().copied().map(vec4).collect(),
            velocities: self.vel.iter().copied().map(vec4).collect(),
            masses: self.mass.iter().map(|&m| m as f32).collect(),
            tet_indices,
            tet_grads,
            tet_volumes,
            tet_mu,
            tet_lambda,
            bead_ids: self.bead_offsets.iter().map(|&(id, _)| id as u32).collect(),
            bead_offsets: self
                .bead_offsets
                .iter()
                .map(|&(_, offset)| vec4(offset))
                .collect(),
            tread_ids: self.tread_nodes.iter().map(|&id| id as u32).collect(),
            pressure_area_per_tread_node: pressure_area_per_tread_node as f32,
            cavity_radial_force_per_bead,
            tet_fiber_dir,
            tet_fiber_k1,
            tet_fiber_k2,
            tet_fiber_dir_b,
            tet_fiber_k1_b,
            tet_fiber_k2_b,
            tet_fiber_dir_c,
            tet_fiber_k1_c,
            tet_fiber_k2_c,
            tet_fiber_dir_d,
            tet_fiber_k1_d,
            tet_fiber_k2_d,
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
        }
    }

    /// Stage (f) ext — apply node positions/velocities AND the 3-DOF
    /// GPU-resolved hub state `[x, y, z, v_y, v_x, v_z, _, _]`.
    pub fn apply_gpu_node_state(
        &mut self,
        positions: &[[f32; 4]],
        velocities: &[[f32; 4]],
        hub: [f32; 8],
    ) -> Result<(), &'static str> {
        if positions.len() != self.pos.len() || velocities.len() != self.vel.len() {
            return Err("gpu tire FEM output length mismatch");
        }
        for (dst, src) in self.pos.iter_mut().zip(positions.iter()) {
            *dst = [src[0] as f64, src[1] as f64, src[2] as f64];
        }
        for (dst, src) in self.vel.iter_mut().zip(velocities.iter()) {
            *dst = [src[0] as f64, src[1] as f64, src[2] as f64];
        }
        self.hub = [hub[0] as f64, hub[1] as f64, hub[2] as f64];
        self.hub_v_y = hub[3] as f64;
        self.hub_v_xz = [hub[4] as f64, hub[5] as f64];
        Ok(())
    }

    /// Stage (f) ext — read the lateral / longitudinal hub velocity (m/s).
    pub fn hub_lateral_velocity(&self) -> [f64; 2] {
        self.hub_v_xz
    }

    /// Stage T5 — kinematically prescribe the hub pose + velocity. Use
    /// this when the demo drives the tire along a predetermined path
    /// (e.g. a skidpad) rather than integrating the hub from forces.
    ///
    /// PLAN_TOURQUE Stage D — also flips the hub into kinematic mode,
    /// so a later `step_on_terrain` keeps the prescribed pose.
    pub fn set_hub_kinematic(&mut self, hub: [f64; 3], vel_xyz: [f64; 3]) {
        self.hub = hub;
        self.hub_v_y = vel_xyz[1];
        self.hub_v_xz = [vel_xyz[0], vel_xyz[2]];
        self.hub_kinematic = true;
    }

    /// PLAN_TOURQUE Stage D — switch the hub XZ pose into "dynamic"
    /// mode: from this call on, `step_on_terrain` integrates `hub_v_xz`
    /// from the per-substep net contact reaction and advances
    /// `hub[0]` / `hub[2]` from that velocity. This is the path scene
    /// 123 (and any future powertrain-driven tire) takes; everything
    /// else defaults to kinematic to preserve existing behaviour.
    pub fn set_hub_dynamic(&mut self) {
        self.hub_kinematic = false;
    }

    /// PLAN_TOURQUE Stage D — true when the hub is being driven from
    /// contact reaction. Mirrors `set_hub_dynamic` / `set_hub_kinematic`.
    pub fn hub_is_dynamic(&self) -> bool {
        !self.hub_kinematic
    }

    /// PLAN_TOURQUE Stage F — chassis hand-off: prescribe just the hub
    /// pose, leave velocity untouched. The intended use is a
    /// suspension/chassis loop that pushes a fresh target each frame
    /// derived from chassis state. Flips the hub into kinematic mode
    /// (same as `set_hub_kinematic`).
    pub fn set_hub_position(&mut self, p: [f64; 3]) {
        if p.iter().all(|v| v.is_finite()) {
            self.hub = p;
        }
        self.hub_kinematic = true;
    }

    /// PLAN_TOURQUE Stage F — chassis hand-off: prescribe just the hub
    /// velocity, leave pose untouched. Same kinematic semantics as
    /// `set_hub_position` and `set_hub_kinematic`.
    pub fn set_hub_velocity(&mut self, v: [f64; 3]) {
        if v.iter().all(|c| c.is_finite()) {
            self.hub_v_xz = [v[0], v[2]];
            self.hub_v_y = v[1];
        }
        self.hub_kinematic = true;
    }

    /// PLAN_TOURQUE Stage A — set the drive (acceleration) torque applied
    /// at the wheel hub axis, in newton-metres. Signed: positive values
    /// accelerate `hub_omega` toward +ω (forward rolling under the sign
    /// convention pinned in PLAN_TOURQUE.md).
    ///
    /// Stage A only stores the value; Stage B integrates it into spin;
    /// Stages C/D couple spin to contact friction and translate the hub.
    pub fn set_drive_torque_n_m(&mut self, n_m: f64) {
        self.drive_torque_n_m = if n_m.is_finite() { n_m } else { 0.0 };
    }

    /// PLAN_TOURQUE Stage A — set the brake torque magnitude applied at
    /// the wheel hub axis, in newton-metres. Always interpreted as a
    /// non-negative value that opposes the current `hub_omega`.
    pub fn set_brake_torque_n_m(&mut self, n_m: f64) {
        self.brake_torque_n_m = if n_m.is_finite() { n_m.max(0.0) } else { 0.0 };
    }

    /// Set the wheel spin angular velocity directly (rad/s, signed;
    /// +ω = forward rolling). Use this to start a tire already rolling —
    /// vehicles hand wheels over with a nonzero road speed, and parity /
    /// rig fixtures need significant contact slip from the first substep
    /// instead of waiting for drive torque to spin the wheel up.
    pub fn set_hub_spin(&mut self, omega_rad_per_s: f64) {
        self.hub_omega = if omega_rad_per_s.is_finite() {
            omega_rad_per_s
        } else {
            0.0
        };
    }

    /// PLAN_TOURQUE Stage A — current drive torque setpoint (N·m).
    pub fn drive_torque_n_m(&self) -> f64 {
        self.drive_torque_n_m
    }

    /// PLAN_TOURQUE Stage A — current brake torque setpoint (N·m).
    pub fn brake_torque_n_m(&self) -> f64 {
        self.brake_torque_n_m
    }

    /// PLAN_TOURQUE Stage A — current spin angle around the wheel axis
    /// (radians). Stage B drives this from torque.
    pub fn hub_angle(&self) -> f64 {
        self.hub_angle
    }

    /// Render-time rotated node positions (the BeamNG/AC/iRacing
    /// pattern): physics state stays in the non-rotating tire frame and
    /// the visible rolling is applied only here, by rigidly rotating the
    /// deformed mesh about the hub spin axis by `hub_angle`. Rolling
    /// enters the physics through the slip equations, never through node
    /// positions, so this cannot destabilize the solver. Trade-off
    /// (documented in PROGRESS.md): the deformation rotates with the
    /// mesh instead of staying glued to world-bottom; at typical speeds
    /// elastic relaxation makes that read as material flowing through
    /// the contact patch, which is what a real rolling tire looks like.
    pub fn nodes_rendered(&self) -> Vec<[f64; 3]> {
        let (s, c) = self.hub_angle.sin_cos();
        self.pos
            .iter()
            .map(|p| {
                let lx = p[0] - self.hub[0];
                let ly = p[1] - self.hub[1];
                [
                    c * lx + s * ly + self.hub[0],
                    -s * lx + c * ly + self.hub[1],
                    p[2],
                ]
            })
            .collect()
    }

    /// PLAN_TOURQUE Stage A — current spin angular velocity around the
    /// wheel axis (rad/s). Stage B drives this from torque.
    pub fn hub_omega(&self) -> f64 {
        self.hub_omega
    }

    /// PLAN_TOURQUE Stage A — most-recent net wrench at the hub (N).
    /// Stage D writes the integrated contact reaction here; until then
    /// this remains zero.
    pub fn last_hub_reaction_xyz(&self) -> [f64; 3] {
        self.hub_reaction_xyz
    }

    /// PLAN_TOURQUE Stage B — rigid-equivalent wheel moment of inertia
    /// about the spin axis (kg·m²), computed from the FEM node masses
    /// and the rest outer radius. Used as `I_wheel` in the
    /// `α = τ / I` integration in `integrate_spin_dof`.
    pub fn wheel_moment_of_inertia(&self) -> f64 {
        let total_node_mass: f64 = self.mass.iter().sum();
        let r = self.params.outer_radius.max(1.0e-3);
        // Thin-walled-cylinder equivalent: I ≈ m · r². This is an upper
        // bound for a hollow tire; refine later if the FEM mass
        // distribution diverges (see PLAN_TOURQUE.md Stage B notes).
        (total_node_mass + self.params.hub_mass.max(0.0)) * r * r
    }

    /// PLAN_ROLL Stage C phase 1 — apply a rigid rotation by `theta`
    /// (radians) about the hub's +Z spin axis to every piece of the
    /// FEM state that lives in the body frame. Specifically:
    ///
    ///   * `pos[i]` and `rest[i]` rotate around the hub (XY).
    ///   * `vel[i]` rotates around the origin (no hub translation for
    ///     velocities — only the rotation).
    ///   * Each bead's offset-from-hub rotates around the origin.
    ///   * Each tet's shape-function gradients rotate so the
    ///     deformation gradient F = sum p_i ⊗ g_i picks up the same R
    ///     consistently.
    ///   * Each tet's fibre-direction families rotate so the anisotropic
    ///     stress acts along the now-rotated material axis.
    ///
    /// `dihedral`, `tread_belt_edges`, and `sidewall_bend_edges` rest
    /// lengths are scalars and rotation-invariant — they need no
    /// update. Same for the cavity gas volume.
    #[allow(dead_code)] // PLAN_ROLL Stage C proper will re-enable this.
    fn apply_rigid_rotation_about_hub(&mut self, theta: f64) {
        if !theta.is_finite() || theta == 0.0 {
            return;
        }
        let (s, c) = theta.sin_cos();
        let hx = self.hub[0];
        let hy = self.hub[1];
        // pos and rest rotate around the hub.
        //
        // NOTE: we intentionally do NOT rotate `self.vel` by the same
        // R. Velocities are real world-frame motions integrated from
        // forces — rotating them per-substep mixes contact-induced
        // velocity components (e.g. the upward Δv from an XPBD
        // pop-out) back into the tangential direction, the next
        // substep's contact code corrects, the cycle repeats, and
        // the visible result is a jelly-like back-and-forth wobble
        // on the upper carcass. Keeping `vel` in its natural frame
        // lets the integrator + contact branch keep the velocities
        // consistent with the rotated positions on their own.
        for i in 0..self.pos.len() {
            let lx = self.pos[i][0] - hx;
            let ly = self.pos[i][1] - hy;
            self.pos[i][0] = c * lx - s * ly + hx;
            self.pos[i][1] = s * lx + c * ly + hy;
            let rlx = self.rest[i][0] - hx;
            let rly = self.rest[i][1] - hy;
            self.rest[i][0] = c * rlx - s * rly + hx;
            self.rest[i][1] = s * rlx + c * rly + hy;
        }
        // Bead offsets are hub-relative vectors.
        for (_, offset) in self.bead_offsets.iter_mut() {
            let ox = offset[0];
            let oy = offset[1];
            offset[0] = c * ox - s * oy;
            offset[1] = s * ox + c * oy;
        }
        // Tetra shape gradients rotate the same way the body-frame
        // vectors they describe do, so `F = sum p_i ⊗ g_i` picks up a
        // consistent R on both sides.
        for tet in self.tets.iter_mut() {
            for g in tet.grads.iter_mut() {
                let gx = g[0];
                let gy = g[1];
                g[0] = c * gx - s * gy;
                g[1] = s * gx + c * gy;
            }
            // Fibre directions rotate too — they're body-frame unit
            // vectors, and the rotated material axis is where the
            // fibre stress should now act.
            for dir in [
                &mut tet.fiber_dir,
                &mut tet.fiber_dir_b,
                &mut tet.fiber_dir_c,
                &mut tet.fiber_dir_d,
            ] {
                let dx = dir[0];
                let dy = dir[1];
                dir[0] = c * dx - s * dy;
                dir[1] = s * dx + c * dy;
            }
        }
    }

    /// PLAN_TOURQUE Stage B — advance the spin DOF for one substep.
    ///
    /// `hub_omega += (drive_torque − brake_torque · soft_sign(omega)) / I · dt`,
    /// then `hub_angle += hub_omega · dt`. Brake torque is folded through
    /// a saturating opposing term so it cannot make the spin jump past
    /// zero in a single substep.
    fn integrate_spin_dof(&mut self, dt: f64) {
        if !(dt.is_finite() && dt > 0.0) {
            return;
        }
        let i_wheel = self.wheel_moment_of_inertia().max(1.0e-6);
        let omega_abs = self.hub_omega.abs();
        // Soft-sign saturator: vanishes near omega=0 so the brake cannot
        // overshoot through zero in one substep, and approaches ±1 at
        // higher spin. The 0.5 rad/s shoulder is well below typical
        // rolling-tire spin (a 0.32 m wheel at 1 m/s rolls at 3.1 rad/s).
        let brake_soft = self.hub_omega / (omega_abs + 0.5);
        let net_torque = self.drive_torque_n_m - self.brake_torque_n_m * brake_soft;
        let alpha = net_torque / i_wheel;
        self.hub_omega += alpha * dt;
        if !self.hub_omega.is_finite() {
            self.hub_omega = 0.0;
        }
        self.hub_angle += self.hub_omega * dt;
        if !self.hub_angle.is_finite() {
            self.hub_angle = 0.0;
        }
    }

    /// Stage T5 — update the per-substep vertical load (N) applied at
    /// the hub. Lets a demo slider drive load live without rebuilding
    /// the tire each change.
    pub fn set_downforce(&mut self, n: f64) {
        self.params.downforce = n.max(0.0);
    }

    /// Stage T5 — current downforce setpoint (N).
    pub fn downforce(&self) -> f64 {
        self.params.downforce
    }

    /// Stage T5 — read the hub's vertical deflection from its at-rest
    /// position. Positive = the rim has sunk under load (m).
    pub fn hub_deflection_m(&self) -> f64 {
        let rest_hub_y = self.params.outer_radius * 1.05;
        (rest_hub_y - self.hub[1]).max(0.0)
    }

    pub fn apply_gpu_node_step(
        &mut self,
        positions: &[[f32; 4]],
        velocities: &[[f32; 4]],
        dt: f64,
    ) -> Result<(), &'static str> {
        if positions.len() != self.pos.len() || velocities.len() != self.vel.len() {
            return Err("gpu tire FEM output length mismatch");
        }
        for (dst, src) in self.pos.iter_mut().zip(positions.iter()) {
            *dst = [src[0] as f64, src[1] as f64, src[2] as f64];
        }
        for (dst, src) in self.vel.iter_mut().zip(velocities.iter()) {
            *dst = [src[0] as f64, src[1] as f64, src[2] as f64];
        }

        // The first GPU kernel updates nodes. Keep the one-DOF hub integration
        // on CPU by recomputing the equal/opposite bead reaction from the
        // updated bead-node positions. This keeps the viewer fully using the
        // GPU node FEM while preserving the loaded hub motion until the GPU
        // reduction path lands.
        let mut hub_force_y = -self.params.downforce - 9.81 * self.params.hub_mass;
        for &(id, offset) in &self.bead_offsets {
            let target = add(self.hub, offset);
            let delta = sub(target, self.pos[id]);
            let rel_v_y = self.hub_v_y - self.vel[id][1];
            let f_y = self.params.bead_stiffness * delta[1] + self.params.bead_damping * rel_v_y;
            hub_force_y -= f_y;
        }
        hub_force_y -= 260.0 * self.hub_v_y;
        self.hub_v_y += hub_force_y / self.params.hub_mass * dt;
        self.hub_v_y = self.hub_v_y.clamp(-2.5, 2.5);
        self.hub[1] += self.hub_v_y * dt;
        let hub_floor = self.params.inner_radius * 0.75;
        if self.hub[1] < hub_floor {
            self.hub[1] = hub_floor;
            if self.hub_v_y < 0.0 {
                self.hub_v_y = 0.0;
            }
        }
        Ok(())
    }

    pub fn step(&mut self, dt: f64) {
        let terrain = FlatTerrain::new(0.0, self.params.surface_material.mu_static);
        self.step_on_terrain(&terrain, dt);
    }

    /// Step the FEM tire against a terrain query instead of the default flat
    /// floor. This is the CPU path for heightfield/material/moving-ground
    /// contact feedback into FEM nodes; GPU terrain sampling remains a follow-
    /// up kernel path.
    pub fn step_on_terrain(&mut self, terrain: &dyn Terrain, dt: f64) {
        match self.params.solver_mode {
            TetraTireFemSolverMode::ExplicitDefault => {
                // The floor-penalty CFL is now gone (see `step_inner` — contact is
                // an XPBD positional projection). The remaining `max_dt` budget is
                // set by the **explicit Neo-Hookean** elastic step: at E=8 MPa for
                // the tread and ~5 mm element size, the elastic CFL is ≈2e-4 s,
                // and low-pressure tires (no membrane-stiffening boost) lose
                // their stability margin first. Stepping beyond that manifests as
                // a tire that "stalls" before forming a proper patch — caught by
                // the `tetra_fem_pressure_reduces_patch_width` test. Implicit
                // Neo-Hookean is the next stage to unlock a larger `dt`.
                let max_dt = 1.25e-4;
                let substeps = (dt / max_dt).ceil().max(1.0) as usize;
                let h = dt / substeps as f64;
                for _ in 0..substeps {
                    self.step_inner(h, terrain);
                }
            }
            TetraTireFemSolverMode::MatrixFreeNewtonDebug => {
                // Production-promotion seam for the matrix-free Newton work.
                // Still labelled CPU/debug in the GPU parity ledger — see the
                // `tetra_fem_newton_solver_debug` row. Returning a report from
                // here would change the trait signature, so we just discard it
                // when the caller opted in through `solver_mode`.
                let _ = self.step_on_terrain_matrix_free_newton_debug(terrain, dt);
            }
        }
    }

    /// CPU/debug matrix-free implicit tire step.
    ///
    /// This is the first callable timestep path using the matrix-free Newton
    /// plumbing from `TetraFemMatrixFreeNewtonStepReport`. It is intentionally
    /// opt-in and still recorded as CPU/debug in the parity ledger: production
    /// viewer stepping keeps using [`Self::step_on_terrain`] until this path
    /// has GPU sparse/matrix-free parity and broader live-scene validation.
    pub fn step_on_terrain_matrix_free_newton_debug(
        &mut self,
        terrain: &dyn Terrain,
        dt: f64,
    ) -> TetraFemMatrixFreeNewtonStepReport {
        let dt_s = finite_positive_or(dt, 1.0 / 300.0, 1.0e-8).min(1.0 / 60.0);
        let node_count = self.pos.len();
        let x0 = self.pos.clone();
        let v0 = self.vel.clone();
        let masses = self.mass.clone();
        let tets = self.tets.clone();
        let node_damping = self.params.node_damping;
        let downforce = self.params.downforce.max(0.0);
        let mut external = vec![[0.0; 3]; node_count];
        for (id, force) in external.iter_mut().enumerate() {
            force[1] -= 9.81 * masses[id].max(1.0e-9);
            force[0] += self.external_force_xyz[0] / node_count.max(1) as f64;
            force[1] += self.external_force_xyz[1] / node_count.max(1) as f64;
            force[2] += self.external_force_xyz[2] / node_count.max(1) as f64;
        }
        let tread_share = self.tread_nodes.len().max(1) as f64;
        for &id in &self.tread_nodes {
            external[id][1] -= downforce / tread_share;
        }
        let bead_share = self.bead_offsets.len().max(1) as f64;
        for &(id, _) in &self.bead_offsets {
            external[id][1] += downforce / bead_share;
        }

        let outcome =
            matrix_free_newton_correction(&tets, &x0, &v0, &masses, node_damping, &external, dt_s);

        self.pos = patch_unflatten_nodes(&outcome.best_x);
        for (i, (p, p0)) in self.pos.iter().zip(x0.iter()).enumerate() {
            self.vel[i] = scale(sub(*p, *p0), 1.0 / dt_s);
            for axis in 0..3 {
                self.vel[i][axis] = finite_or(self.vel[i][axis], 0.0).clamp(-12.0, 12.0);
            }
        }
        self.project_structural_constraints(dt_s, terrain);
        self.project_volume_constraints(dt_s, terrain);
        self.project_nodes_against_terrain(terrain);

        let corrected_x = patch_flatten_nodes(&self.pos);
        let corrected_residual_norm = vec_norm(&patch_newton_residual_with_masses(
            &tets,
            &x0,
            &v0,
            &corrected_x,
            &masses,
            node_damping,
            &external,
            dt_s,
        ));
        let mut max_velocity_m_per_s = 0.0_f64;
        for v in &self.vel {
            max_velocity_m_per_s = max_velocity_m_per_s.max(dot(*v, *v).sqrt());
        }
        let mut min_volume_ratio = f64::INFINITY;
        let mut max_volume_ratio = 0.0_f64;
        for tet in &self.tets {
            let ratio = tetra_signed_volume(tet.ids, &self.pos).abs() / tet.volume.max(1.0e-30);
            min_volume_ratio = min_volume_ratio.min(ratio);
            max_volume_ratio = max_volume_ratio.max(ratio);
        }
        let explicit_energy_j =
            patch_total_energy_with_masses(&self.tets, &x0, &outcome.predictor, &self.mass, dt_s);
        let implicit_energy_j =
            patch_total_energy_with_masses(&self.tets, &x0, &self.pos, &self.mass, dt_s);
        TetraFemMatrixFreeNewtonStepReport {
            node_count,
            tetra_count: self.tets.len(),
            coupled_dof_count: outcome.dof,
            dt_s,
            initial_residual_norm: outcome.initial_residual_norm,
            corrected_residual_norm,
            residual_reduction_ratio: corrected_residual_norm
                / outcome.initial_residual_norm.max(1.0e-30),
            line_search_alpha: outcome.best_alpha,
            krylov_iteration_count: outcome.krylov_iteration_count,
            diagonal_jv_evaluation_count: outcome.dof,
            correction_norm_m: vec_norm(&outcome.correction),
            max_correction_m: vec_max_abs(&outcome.correction),
            max_velocity_m_per_s,
            min_volume_ratio,
            max_volume_ratio,
            explicit_energy_j,
            implicit_energy_j,
            energy_ratio_implicit_over_explicit: implicit_energy_j / explicit_energy_j.max(1.0e-30),
            finite_outputs: outcome.initial_residual_norm.is_finite()
                && corrected_residual_norm.is_finite()
                && outcome.diagonal.iter().all(|value| value.is_finite())
                && outcome.dx.iter().all(|value| value.is_finite())
                && self.pos.iter().all(|p| finite_vec3(*p))
                && self.vel.iter().all(|v| finite_vec3(*v)),
        }
    }

    pub fn contact_patch_width(&self) -> f64 {
        let xs: Vec<f64> = self
            .tread_nodes
            .iter()
            .map(|&id| self.pos[id])
            .filter(|p| p[1] <= 3.0e-2)
            .map(|p| p[0])
            .collect();
        if xs.len() < 2 {
            return 0.0;
        }
        let mn = xs.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        mx - mn
    }

    /// Evaluate wheel-level brush contact from the **actual deformed FEM tread
    /// nodes**, not from an ideal circular sample patch.
    ///
    /// This is the first CPU path for PLAN_REALTIRE FEM/contact coupling: the
    /// active contact cells come from current tread-node world positions, local
    /// node velocities contribute to slip, and the reported patch diagnostics
    /// reflect the deformed tire shape. The FEM integrator still has its own
    /// node-level contact projection; feeding this wheel-level brush wrench
    /// back into the FEM nodes is the next deeper coupling step.
    pub fn evaluate_tread_contact(
        &self,
        state: &RealWheelState,
        terrain: &dyn Terrain,
        material: &SurfaceMaterial,
    ) -> RealTireContact {
        if !finite_vec3(state.hub_pos) || !finite_vec3(state.hub_vel) || !finite_vec3(state.heading)
        {
            return RealTireContact::empty();
        }
        let material = sanitized_surface_material(material);
        let normal = normalize(terrain.normal(state.hub_pos[0], state.hub_pos[2]));
        let heading = normalize(project_to_ground(state.heading, normal));
        let side = normalize(cross(heading, normal));
        let omega = finite_or(state.omega, 0.0);
        let area_node = self.tread_area_per_node.max(1.0e-6);
        let pressure_patch_factor = (180_000.0 / self.params.pressure_pa.max(20_000.0))
            .sqrt()
            .clamp(0.65, 1.75);
        let contact_band =
            (self.params.outer_radius * 0.10 * pressure_patch_factor).clamp(0.012, 0.060);
        let target_normal_load = self.params.downforce.max(0.0);
        let hub_vel_internal = [self.hub_v_xz[0], self.hub_v_y, self.hub_v_xz[1]];

        let terrain_mu_ref =
            finite_positive_or(terrain.friction(), material.mu_static.max(1.0e-3), 1.0e-6);
        let mut candidates: Vec<(usize, f64, f64, f64, f64, f64, f64)> = Vec::new();
        let mut weight_sum = 0.0;
        for &id in &self.tread_nodes {
            let p = self.pos[id];
            let ground_y = terrain.height(p[0], p[2]);
            let clearance = p[1] - ground_y;
            if clearance > contact_band {
                continue;
            }
            let weight = ((contact_band - clearance) / contact_band).clamp(0.0, 3.0);
            if weight <= 0.0 {
                continue;
            }
            let local_x = dot(sub(p, self.hub), heading);
            let local_z = dot(sub(p, self.hub), side);
            let local_y = dot(sub(p, self.hub), normal);
            candidates.push((id, weight, local_x, local_y, local_z, p[0], p[2]));
            weight_sum += weight;
        }
        if candidates.is_empty() || weight_sum <= 1.0e-12 || target_normal_load <= 0.0 {
            return RealTireContact::empty();
        }

        // Patch extent and travel direction for the linear leading-edge
        // bristle profile: deflection is zero where tread enters contact
        // and grows toward the trailing edge, so longer (lower-pressure)
        // patches develop more tractive force before the friction cap.
        let mut patch_min_x = f64::INFINITY;
        let mut patch_max_x = f64::NEG_INFINITY;
        for candidate in &candidates {
            patch_min_x = patch_min_x.min(candidate.2);
            patch_max_x = patch_max_x.max(candidate.2);
        }
        let patch_extent = (patch_max_x - patch_min_x).max(0.0);
        let hub_travel = dot(state.hub_vel, heading) + omega * self.params.outer_radius;
        let leading_x = if hub_travel >= 0.0 {
            patch_max_x
        } else {
            patch_min_x
        };
        let cell_len =
            self.params.outer_radius * std::f64::consts::TAU / self.params.segments.max(1) as f64;
        let patch_gain =
            brush_pressure_patch_gain(0.0, self.params.pressure_pa, self.params.outer_radius);

        let mut cells = Vec::with_capacity(candidates.len());
        let mut normal_load = 0.0;
        let mut fx = 0.0;
        let mut fy = 0.0;
        let mut mz = 0.0;
        let mut normal_moment_x = 0.0;
        let mut lateral_moment_x = 0.0;
        let mut sliding_count = 0usize;
        let mut available_friction_budget = 0.0;
        let mut consumed_friction_budget = 0.0;
        let mut dissipated_power = 0.0;
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut slip_ratio_sum = 0.0;
        let mut slip_speed_sum = 0.0;
        let mut lateral_slip_speed_sum = 0.0;
        let mut slip_angle_sum = 0.0;
        let mut force_vec = [0.0; 3];
        let mut rolling_resistance_force = 0.0;

        for (id, weight, local_x, local_y, local_z, world_x, world_z) in candidates {
            let fn_i = target_normal_load * weight / weight_sum;
            let node_rel_vel = sub(self.vel[id], hub_vel_internal);
            let cell_normal = normalize(terrain.normal(world_x, world_z));
            let cell_heading = normalize(project_to_ground(state.heading, cell_normal));
            let cell_side = normalize(cross(cell_heading, cell_normal));
            let ground_v = terrain.surface_velocity(world_x, world_z);
            let node_world_vel = sub(add(state.hub_vel, node_rel_vel), ground_v);
            let v_long = dot(node_world_vel, cell_heading);
            let v_lat = dot(node_world_vel, cell_side);
            let rolling_radius = (-local_y)
                .abs()
                .clamp(self.params.inner_radius, self.params.outer_radius);
            let wheel_speed = omega * rolling_radius;
            let denom = v_long.abs().max(0.35);
            let slip_speed = wheel_speed - v_long;
            let slip_ratio = (slip_speed / denom).clamp(-8.0, 8.0);
            let lateral_slip_speed = -v_lat;
            let cell_material = surface_material_with_terrain_film(
                &material,
                terrain,
                world_x,
                world_z,
                (slip_speed * slip_speed + lateral_slip_speed * lateral_slip_speed).sqrt(),
            );
            let terrain_mu = finite_nonnegative(terrain.friction_at(world_x, world_z));
            let friction_scale = (terrain_mu / terrain_mu_ref).clamp(0.0, 3.0);
            let mu_static_cell = cell_material.mu_static * friction_scale;
            let mu_dynamic_cell = cell_material.mu_dynamic * friction_scale;
            let slip_angle_rad = v_lat.atan2(denom).clamp(-1.2, 1.2);
            let normal_pressure = fn_i / area_node;
            let (load_mu_factor, load_stiffness_factor) = brush_load_sensitivity(normal_pressure);
            let shear_k_eff = self.params.tread_shear_stiffness
                * load_stiffness_factor
                * brush_pressure_stiffness_factor(self.params.pressure_pa);
            let profile =
                patch_gain * brush_bristle_profile(local_x, leading_x, patch_extent, cell_len);
            let bristle_disp_long = slip_ratio * rolling_radius * profile;
            let bristle_disp_lat = lateral_slip_speed / denom * rolling_radius * profile;
            let trial_long = shear_k_eff * area_node * bristle_disp_long
                + cell_material.bristle_damping * area_node * slip_speed;
            let trial_lat = shear_k_eff * area_node * bristle_disp_lat
                + cell_material.bristle_damping * area_node * lateral_slip_speed;
            let static_cap =
                mu_static_cell * load_mu_factor * fn_i + cell_material.cohesion * area_node;
            let dynamic_cap =
                mu_dynamic_cell * load_mu_factor * fn_i + cell_material.cohesion * area_node;
            let (ft_long, ft_lat, sliding, available_i, consumed_i) =
                smooth_combined_brush_force(trial_long, trial_lat, static_cap, dynamic_cap);
            let rolling_i = cell_material.rolling_resistance * fn_i * v_long.signum();
            rolling_resistance_force += rolling_i;

            normal_load += fn_i;
            fx += ft_long;
            fy += ft_lat;
            force_vec = add(
                force_vec,
                add(
                    add(
                        scale(cell_heading, ft_long - rolling_i),
                        scale(cell_side, ft_lat),
                    ),
                    scale(cell_normal, fn_i),
                ),
            );
            mz -= local_x * ft_lat;
            normal_moment_x += local_x * fn_i;
            lateral_moment_x += local_x * ft_lat.abs();
            available_friction_budget += available_i;
            consumed_friction_budget += consumed_i;
            min_x = min_x.min(local_x);
            max_x = max_x.max(local_x);
            slip_ratio_sum += slip_ratio;
            slip_speed_sum += slip_speed;
            lateral_slip_speed_sum += lateral_slip_speed;
            slip_angle_sum += slip_angle_rad;
            if sliding {
                sliding_count += 1;
                dissipated_power += (ft_long * ft_long + ft_lat * ft_lat).sqrt()
                    * (slip_speed * slip_speed + lateral_slip_speed * lateral_slip_speed).sqrt();
            }
            cells.push(TireContactCell {
                node_index: Some(id),
                local_x,
                local_z,
                penetration: weight * contact_band,
                area: area_node,
                normal_load: fn_i,
                normal: cell_normal,
                tangential_force: ft_long,
                lateral_force: ft_lat,
                sliding,
            });
        }

        let rolling = rolling_resistance_force;
        let longitudinal_force = fx - rolling;
        let n = cells.len().max(1) as f64;
        let contact_center_x = if normal_load > 1.0e-9 {
            normal_moment_x / normal_load
        } else {
            0.0
        };
        let force_resultant_x = if fy.abs() > 1.0e-9 {
            lateral_moment_x / fy.abs()
        } else {
            contact_center_x
        };
        let pneumatic_trail_m = contact_center_x - force_resultant_x;
        let slip_ratio = slip_ratio_sum / n;
        let slip_speed = slip_speed_sum / n;
        let lateral_slip_speed = lateral_slip_speed_sum / n;
        let slip_angle_rad = slip_angle_sum / n;
        let sliding_fraction = sliding_count as f64 / n;
        let adhesion_fraction = (1.0 - sliding_fraction).clamp(0.0, 1.0);
        let effective_longitudinal_stiffness = {
            let disp = (slip_ratio * self.params.outer_radius).abs();
            if disp > 1.0e-6 { fx.abs() / disp } else { 0.0 }
        };
        let effective_cornering_stiffness = if slip_angle_rad.abs() > 1.0e-5 {
            fy.abs() / slip_angle_rad.abs()
        } else {
            0.0
        };
        let force = force_vec;
        RealTireContact {
            in_contact: true,
            force,
            normal_load,
            longitudinal_force,
            lateral_force: fy,
            aligning_moment_z: mz,
            camber_angle_rad: 0.0,
            contact_center_x,
            force_resultant_x,
            pneumatic_trail_m,
            rolling_resistance_force: rolling,
            slip_ratio,
            slip_speed,
            lateral_slip_speed,
            slip_angle_rad,
            patch_length: (max_x - min_x).max(0.0),
            contact_count: cells.len(),
            sliding_count,
            adhesion_fraction,
            sliding_fraction,
            available_friction_budget,
            consumed_friction_budget,
            effective_longitudinal_stiffness,
            effective_cornering_stiffness,
            cells,
            dissipated_power,
        }
    }

    /// Evaluate FEM-derived tread contact and feed the resulting distributed
    /// wheel-level contact wrench back into the tire nodes as impulses.
    ///
    /// `evaluate_tread_contact` is a readout/evaluation path. This method is
    /// the next coupling step: every returned `TireContactCell` carries the FEM
    /// tread node that produced it, and the cell's normal/longitudinal/lateral
    /// force is applied to that node's velocity. Bead/rim constraints then
    /// transfer the reaction into the hub during following FEM steps.
    ///
    /// This is still CPU-only, but it now uses the tire's persistent
    /// per-tread-node bristle state rather than a stateless trial force. GPU
    /// contact reductions remain the next deeper coupling work.
    pub fn apply_tread_contact_wrench(
        &mut self,
        state: &RealWheelState,
        terrain: &dyn Terrain,
        material: &SurfaceMaterial,
        dt: f64,
    ) -> RealTireContact {
        let mut contact = self.evaluate_tread_contact(state, terrain, material);
        let dt = finite_or(dt, 0.0).max(0.0);
        if !contact.in_contact || dt <= 0.0 {
            return contact;
        }

        let material = sanitized_surface_material(material);
        let terrain_mu_ref =
            finite_positive_or(terrain.friction(), material.mu_static.max(1.0e-3), 1.0e-6);
        let hub_vel_internal = [self.hub_v_xz[0], self.hub_v_y, self.hub_v_xz[1]];
        let omega = finite_or(state.omega, 0.0);

        let mut normal_load = 0.0;
        let mut fx = 0.0;
        let mut fy = 0.0;
        let mut mz = 0.0;
        let mut normal_moment_x = 0.0;
        let mut lateral_moment_x = 0.0;
        let mut sliding_count = 0usize;
        let mut available_friction_budget = 0.0;
        let mut consumed_friction_budget = 0.0;
        let mut dissipated_power = 0.0;
        let mut force_vec = [0.0; 3];
        let mut rolling_resistance_force = 0.0;

        for cell in &mut contact.cells {
            let Some(id) = cell.node_index else {
                continue;
            };
            if id >= self.vel.len() {
                continue;
            }

            let cell_normal = normalize(cell.normal);
            let cell_heading = normalize(project_to_ground(state.heading, cell_normal));
            let cell_side = normalize(cross(cell_heading, cell_normal));
            let p = self.pos[id];
            let node_rel_vel = sub(self.vel[id], hub_vel_internal);
            let ground_v = terrain.surface_velocity(p[0], p[2]);
            let node_world_vel = sub(add(state.hub_vel, node_rel_vel), ground_v);
            let v_long = dot(node_world_vel, cell_heading);
            let v_lat = dot(node_world_vel, cell_side);
            let local_y = dot(sub(p, self.hub), cell_normal);
            let rolling_radius = (-local_y)
                .abs()
                .clamp(self.params.inner_radius, self.params.outer_radius);
            let wheel_speed = omega * rolling_radius;
            let slip_speed = wheel_speed - v_long;
            let lateral_slip_speed = -v_lat;
            let cell_material = surface_material_with_terrain_film(
                &material,
                terrain,
                p[0],
                p[2],
                (slip_speed * slip_speed + lateral_slip_speed * lateral_slip_speed).sqrt(),
            );
            let terrain_mu = finite_nonnegative(terrain.friction_at(p[0], p[2]));
            let friction_scale = (terrain_mu / terrain_mu_ref).clamp(0.0, 3.0);
            let mu_static_cell = cell_material.mu_static * friction_scale;
            let mu_dynamic_cell = cell_material.mu_dynamic * friction_scale;
            let normal_pressure = cell.normal_load / cell.area.max(1.0e-9);
            let (load_mu_factor, load_stiffness_factor) = brush_load_sensitivity(normal_pressure);
            let shear_k_eff = self.params.tread_shear_stiffness
                * load_stiffness_factor
                * brush_pressure_stiffness_factor(self.params.pressure_pa);
            if let Some(ti) = self.node_to_tread[id].map(|t| t as usize) {
                let mut bristle_long = dot(self.tread_bristle[ti], cell_heading);
                let mut bristle_lat = dot(self.tread_bristle[ti], cell_side);
                bristle_long += slip_speed * dt;
                bristle_lat += lateral_slip_speed * dt;

                let trial_long = shear_k_eff * cell.area * bristle_long
                    + cell_material.bristle_damping * cell.area * slip_speed;
                let trial_lat = shear_k_eff * cell.area * bristle_lat
                    + cell_material.bristle_damping * cell.area * lateral_slip_speed;
                let static_cap = mu_static_cell * load_mu_factor * cell.normal_load
                    + cell_material.cohesion * cell.area;
                let dynamic_cap = mu_dynamic_cell * load_mu_factor * cell.normal_load
                    + cell_material.cohesion * cell.area;
                let (ft_long, ft_lat, sliding, available_i, consumed_i) =
                    smooth_combined_brush_force(trial_long, trial_lat, static_cap, dynamic_cap);

                if sliding {
                    let bristle_mag =
                        (bristle_long * bristle_long + bristle_lat * bristle_lat).sqrt();
                    let max_bristle = dynamic_cap / (shear_k_eff * cell.area).max(1.0e-9);
                    if bristle_mag > max_bristle && bristle_mag > 1.0e-12 {
                        let s = max_bristle / bristle_mag;
                        bristle_long *= s;
                        bristle_lat *= s;
                    }
                }
                self.tread_bristle[ti] = add(
                    scale(cell_heading, bristle_long),
                    scale(cell_side, bristle_lat),
                );
                cell.tangential_force = ft_long;
                cell.lateral_force = ft_lat;
                cell.sliding = sliding;
                available_friction_budget += available_i;
                consumed_friction_budget += consumed_i;
            }

            let rolling_i = cell_material.rolling_resistance * cell.normal_load * v_long.signum();
            rolling_resistance_force += rolling_i;
            let cell_force = add(
                add(
                    scale(cell_heading, cell.tangential_force - rolling_i),
                    scale(cell_side, cell.lateral_force),
                ),
                scale(cell_normal, cell.normal_load),
            );
            let inv_m = 1.0 / self.mass[id].max(1.0e-9);
            for (axis, component) in cell_force.iter().enumerate() {
                self.vel[id][axis] += component * inv_m * dt;
                self.vel[id][axis] = finite_or(self.vel[id][axis], 0.0).clamp(-12.0, 12.0);
            }

            normal_load += cell.normal_load;
            fx += cell.tangential_force;
            fy += cell.lateral_force;
            mz -= cell.local_x * cell.lateral_force;
            normal_moment_x += cell.local_x * cell.normal_load;
            lateral_moment_x += cell.local_x * cell.lateral_force.abs();
            if cell.sliding {
                sliding_count += 1;
                dissipated_power += (cell.tangential_force * cell.tangential_force
                    + cell.lateral_force * cell.lateral_force)
                    .sqrt()
                    * (slip_speed * slip_speed + lateral_slip_speed * lateral_slip_speed).sqrt();
            }
            force_vec = add(force_vec, cell_force);
        }

        let rolling = rolling_resistance_force;
        let n = contact.cells.len().max(1) as f64;
        let contact_center_x = if normal_load > 1.0e-9 {
            normal_moment_x / normal_load
        } else {
            0.0
        };
        let force_resultant_x = if fy.abs() > 1.0e-9 {
            lateral_moment_x / fy.abs()
        } else {
            contact_center_x
        };
        contact.force = force_vec;
        contact.normal_load = normal_load;
        contact.longitudinal_force = fx - rolling;
        contact.lateral_force = fy;
        contact.aligning_moment_z = mz;
        contact.contact_center_x = contact_center_x;
        contact.force_resultant_x = force_resultant_x;
        contact.pneumatic_trail_m = contact_center_x - force_resultant_x;
        contact.rolling_resistance_force = rolling;
        contact.sliding_count = sliding_count;
        contact.adhesion_fraction = (1.0 - sliding_count as f64 / n).clamp(0.0, 1.0);
        contact.sliding_fraction = (sliding_count as f64 / n).clamp(0.0, 1.0);
        contact.available_friction_budget = available_friction_budget;
        contact.consumed_friction_budget = consumed_friction_budget;
        contact.effective_longitudinal_stiffness = {
            let disp = (contact.slip_ratio * self.params.outer_radius).abs();
            if disp > 1.0e-6 { fx.abs() / disp } else { 0.0 }
        };
        contact.effective_cornering_stiffness = if contact.slip_angle_rad.abs() > 1.0e-5 {
            fy.abs() / contact.slip_angle_rad.abs()
        } else {
            0.0
        };
        contact.dissipated_power = dissipated_power;

        contact
    }

    pub(crate) fn project_volume_constraints(&mut self, dt: f64, terrain: &dyn Terrain) {
        self.last_xpbd_volume_correction_m = 0.0;
        let iterations = self.params.xpbd_volume_iterations;
        if iterations == 0 || dt <= 0.0 || self.tets.is_empty() {
            return;
        }

        let alpha = self.params.xpbd_volume_compliance.max(0.0) / (dt * dt).max(1.0e-12);
        let mut max_correction: f64 = 0.0;
        for _ in 0..iterations {
            let mut correction = vec![[0.0; 3]; self.pos.len()];
            let mut correction_count = vec![0_u32; self.pos.len()];

            for tet in &self.tets {
                let ids = tet.ids;
                let x0 = self.pos[ids[0]];
                let x1 = self.pos[ids[1]];
                let x2 = self.pos[ids[2]];
                let x3 = self.pos[ids[3]];
                if !(finite_vec3(x0) && finite_vec3(x1) && finite_vec3(x2) && finite_vec3(x3)) {
                    continue;
                }

                let volume = tetra_signed_volume(ids, &self.pos);
                if !volume.is_finite() {
                    continue;
                }
                let c = volume - tet.volume;
                if c.abs() <= tet.volume.max(1.0e-12) * 1.0e-4 {
                    continue;
                }

                // ∂V/∂xᵢ for the signed tetra volume. Rest tets are wound to
                // positive volume in `make_tetra_element`, so the constraint
                // `V(x) − V_rest = 0` preserves orientation as well as size.
                let grads = [
                    scale(cross(sub(x3, x1), sub(x2, x1)), 1.0 / 6.0),
                    scale(cross(sub(x2, x0), sub(x3, x0)), 1.0 / 6.0),
                    scale(cross(sub(x3, x0), sub(x1, x0)), 1.0 / 6.0),
                    scale(cross(sub(x1, x0), sub(x2, x0)), 1.0 / 6.0),
                ];

                let mut denom = alpha;
                for a in 0..4 {
                    let inv_m = 1.0 / self.mass[ids[a]].max(1.0e-9);
                    denom += inv_m * dot(grads[a], grads[a]);
                }
                if denom <= 1.0e-18 || !denom.is_finite() {
                    continue;
                }

                let lambda = -c / denom;
                if !lambda.is_finite() {
                    continue;
                }
                for a in 0..4 {
                    let id = ids[a];
                    let inv_m = 1.0 / self.mass[id].max(1.0e-9);
                    let dp = scale(grads[a], inv_m * lambda);
                    correction[id] = add(correction[id], dp);
                    correction_count[id] += 1;
                }
            }

            for i in 0..self.pos.len() {
                if correction_count[i] == 0 {
                    continue;
                }
                let inv_count = 1.0 / correction_count[i] as f64;
                let dp = scale(correction[i], inv_count);
                if !finite_vec3(dp) {
                    continue;
                }
                max_correction = max_correction.max(dot(dp, dp).sqrt());
                self.pos[i] = add(self.pos[i], dp);
                for (axis, component) in dp.iter().enumerate() {
                    self.vel[i][axis] += component / dt;
                    self.vel[i][axis] = finite_or(self.vel[i][axis], 0.0).clamp(-12.0, 12.0);
                }
            }
            self.project_nodes_against_terrain(terrain);
        }
        self.last_xpbd_volume_correction_m = max_correction;
    }

    pub(crate) fn project_structural_constraints(&mut self, dt: f64, terrain: &dyn Terrain) {
        self.last_xpbd_bead_correction_m = 0.0;
        self.last_xpbd_structural_correction_m = 0.0;
        let iterations = self.params.xpbd_structural_iterations;
        if iterations == 0 || dt <= 0.0 {
            return;
        }

        let mut max_bead: f64 = 0.0;
        let mut max_structural: f64 = 0.0;
        for _ in 0..iterations {
            max_bead = max_bead.max(self.project_bead_constraints(dt));
            max_structural = max_structural.max(project_edge_length_constraints(
                &mut self.pos,
                &mut self.vel,
                &self.mass,
                &self.tread_belt_edges,
                self.params.xpbd_tread_belt_compliance,
                dt,
            ));
            max_structural = max_structural.max(project_edge_length_constraints(
                &mut self.pos,
                &mut self.vel,
                &self.mass,
                &self.sidewall_bend_edges,
                self.params.xpbd_sidewall_bend_compliance,
                dt,
            ));
            max_structural = max_structural.max(project_sidewall_dihedral_constraints(
                &mut self.pos,
                &mut self.vel,
                &self.mass,
                &self.sidewall_dihedrals,
                self.params.xpbd_sidewall_dihedral_compliance,
                dt,
            ));
            max_structural = max_structural.max(project_sidewall_ply_strain_constraints(
                &mut self.pos,
                &mut self.vel,
                &self.mass,
                &self.tets,
                &self.rest,
                self.hub,
                1.35,
                self.params.xpbd_sidewall_dihedral_compliance,
                dt,
            ));
            self.project_nodes_against_terrain(terrain);
        }
        self.last_xpbd_bead_correction_m = max_bead;
        self.last_xpbd_structural_correction_m = max_structural;
    }

    fn project_bead_constraints(&mut self, dt: f64) -> f64 {
        if self.params.xpbd_bead_compliance.is_infinite() || dt <= 0.0 {
            return 0.0;
        }
        let alpha = self.params.xpbd_bead_compliance.max(0.0) / (dt * dt).max(1.0e-12);
        let mut max_correction: f64 = 0.0;
        for &(id, offset) in &self.bead_offsets {
            let target = add(self.hub, offset);
            let delta = sub(self.pos[id], target);
            if !finite_vec3(delta) {
                continue;
            }
            let inv_m = 1.0 / self.mass[id].max(1.0e-9);
            let denom = inv_m + alpha;
            if denom <= 1.0e-18 || !denom.is_finite() {
                continue;
            }
            let dp = scale(delta, -inv_m / denom);
            max_correction = max_correction.max(dot(dp, dp).sqrt());
            self.pos[id] = add(self.pos[id], dp);
            for (axis, component) in dp.iter().enumerate() {
                self.vel[id][axis] += component / dt;
                self.vel[id][axis] = finite_or(self.vel[id][axis], 0.0).clamp(-12.0, 12.0);
            }
        }
        max_correction
    }

    pub(crate) fn project_nodes_against_terrain(&mut self, terrain: &dyn Terrain) {
        for i in 0..self.pos.len() {
            let p = self.pos[i];
            let (gap, raw_normal) = terrain.signed_distance_and_normal(p);
            let normal = normalize(raw_normal);
            if gap < 0.0 {
                self.pos[i] = add(p, scale(normal, -gap));
                let inward_v = dot(self.vel[i], normal);
                if inward_v < 0.0 {
                    self.vel[i] = sub(self.vel[i], scale(normal, inward_v));
                }
            }
        }
    }

    pub(crate) fn step_inner(&mut self, dt: f64, terrain: &dyn Terrain) {
        // Stage-(a)/(e) integrator: semi-implicit Euler with **implicit
        // node damping** plus an **XPBD-style hard contact projection**
        // (zero-gap Lagrange). No more explicit floor penalty spring; no
        // more `c·v` damping term riding inside `force[i]` (the explicit
        // form was the dominant numerical-stiffness term and forced the
        // tiny `max_dt`).
        // PLAN_REALTIRE Stream B Stage C.6 — reuse persistent scratch
        // buffers across substeps instead of re-allocating two `Vec`s
        // per call. At 8 000 substeps/s × 40 wheels this is 320 k
        // allocs/s removed from the hot path. Use `mem::take` to move
        // the buffers out of `self` for the duration of `step_inner`,
        // then put them back at the end. This sidesteps the borrow
        // checker — we can call `&mut self` methods freely while the
        // scratch buffers are owned locally.
        let node_count = self.pos.len();
        let mut force = std::mem::take(&mut self.scratch_force);
        force.clear();
        force.resize(node_count, [0.0; 3]);
        let mut node_stiffness_diag = std::mem::take(&mut self.scratch_node_stiffness_diag);
        node_stiffness_diag.clear();
        node_stiffness_diag.resize(node_count, 0.0);
        let mut hub_force_y = -self.params.downforce - 9.81 * self.params.hub_mass;
        // PLAN_TOURQUE Stage B — integrate the spin DOF from drive/brake
        // torque before the FEM forces are accumulated, so downstream
        // stages (C: contact coupling, D: hub XZ integration) can read a
        // freshly updated `hub_omega` / `hub_angle` for this substep.
        self.integrate_spin_dof(dt);
        // PLAN_ROLL Stage C phase 1 — REVERTED.
        //
        // The rigid-rotation pass produced visible jelly-wobble: the
        // contact branch clamps `v.y >= 0` to prevent re-penetration;
        // the next substep's rotation transforms that clean Y velocity
        // into mixed X/Y components; the next contact branch clamps Y
        // back; the cycle visibly oscillates the upper carcass like a
        // jelly. The proper fix is body-frame canonical storage
        // (PLAN_ROLL Stage C proper) — pos/vel live in the body frame,
        // contact applies world-frame impulses via the body↔world
        // transform. Until that lands the FEM mesh stays in
        // PLAN_TOURQUE state: rigid, translates with the hub from
        // contact reaction but does not visibly rotate. The rolling
        // motion is conveyed by the viewer's spoke gizmos.
        // PLAN_TOURQUE Stage D — accumulate per-substep contact reaction
        // so the dynamic-hub integrator below sees the net wrench from
        // every tread node in contact this step.
        let mut net_contact_fx = 0.0_f64;
        let mut net_contact_fz = 0.0_f64;

        // Gravity only. Damping is folded into the velocity update below.
        for i in 0..self.pos.len() {
            force[i][1] -= 9.81 * self.mass[i];
        }

        // Stage (b) leftover safety: a tiny isotropic stiffening term
        // proportional to pressure. With the anisotropic Holzapfel belt
        // from stage (c) carrying the bulk of the cavity hoop tension,
        // this can be very small — it's just numerical robustness so the
        // carcass under high pressure can't reach the Neo-Hookean
        // log-singularity (J → 0) before the belt kicks in.
        let belt_proxy_mu = 0.0_f64; // Stage (c) — Holzapfel belt carries this.
        for tet in &self.tets {
            // Compressible Neo-Hookean tetra: P = μ(F - F^-T) + λ ln(J) F^-T.
            let mut f_def = [[0.0; 3]; 3];
            for a in 0..4 {
                let id = tet.ids[a];
                for row in 0..3 {
                    for col in 0..3 {
                        f_def[row][col] += self.pos[id][row] * tet.grads[a][col];
                    }
                }
            }
            let Some(inv_f) = inv3(f_def) else {
                continue;
            };
            let f_inv_t = transpose3(inv_f);
            let j = det3(f_def).max(0.20);
            let ln_j = j.ln();
            let mu = tet.mu + belt_proxy_mu;
            let lambda = tet.lambda;
            let mut p_stress = [[0.0; 3]; 3];
            for r in 0..3 {
                for c in 0..3 {
                    p_stress[r][c] =
                        mu * (f_def[r][c] - f_inv_t[r][c]) + lambda * ln_j * f_inv_t[r][c];
                }
            }

            // Stage (c/c2) — anisotropic Holzapfel fibre stress for
            // explicit crossed ply families. Active only in tension
            // (`I₄ > 1`): real belt cords buckle in compression.
            add_holzapfel_fiber_stress(
                &mut p_stress,
                f_def,
                tet.fiber_dir,
                tet.fiber_k1,
                tet.fiber_k2,
            );
            add_holzapfel_fiber_stress(
                &mut p_stress,
                f_def,
                tet.fiber_dir_b,
                tet.fiber_k1_b,
                tet.fiber_k2_b,
            );
            add_holzapfel_fiber_stress(
                &mut p_stress,
                f_def,
                tet.fiber_dir_c,
                tet.fiber_k1_c,
                tet.fiber_k2_c,
            );
            add_holzapfel_fiber_stress(
                &mut p_stress,
                f_def,
                tet.fiber_dir_d,
                tet.fiber_k1_d,
                tet.fiber_k2_d,
            );

            for a in 0..4 {
                let f = scale(mat_vec(p_stress, tet.grads[a]), -tet.volume);
                force[tet.ids[a]] = add(force[tet.ids[a]], f);
                let grad_norm2 = dot(tet.grads[a], tet.grads[a]);
                let fibre_tangent = tet.fiber_k1.abs() * (1.0 + tet.fiber_k2.abs())
                    + tet.fiber_k1_b.abs() * (1.0 + tet.fiber_k2_b.abs())
                    + tet.fiber_k1_c.abs() * (1.0 + tet.fiber_k2_c.abs())
                    + tet.fiber_k1_d.abs() * (1.0 + tet.fiber_k2_d.abs());
                node_stiffness_diag[tet.ids[a]] +=
                    tet.volume * (mu.abs() + lambda.abs() + fibre_tangent) * grad_norm2;
            }
        }

        // Bead/rim coupling: the innermost volumetric layer is attached to
        // the wheel hub, and equal/opposite Y force loads the hub.
        for &(id, offset) in &self.bead_offsets {
            let target = add(self.hub, offset);
            let delta = sub(target, self.pos[id]);
            let rel_v_y = self.hub_v_y - self.vel[id][1];
            let mut f = scale(delta, self.params.bead_stiffness);
            f[1] += self.params.bead_damping * rel_v_y;
            force[id] = add(force[id], f);
            node_stiffness_diag[id] += self.params.bead_stiffness.max(0.0);
            hub_force_y -= f[1];
        }

        // ── Cavity gas-law pressure traction ─────────────────────────
        //
        // The tire's air pocket lives between the rim and the bead layer.
        // For a closed sealed cavity with N moles of gas, a polytropic
        // process gives `P · V^γ = const`, so the *effective* pressure at
        // any deformation is `P_now = P_ref · (V_ref / V_now)^γ`. With
        // the cavity-surface triangulation built in `new()`, we apply
        // `F = P_now · n · A` to every cavity facet, sharing 1/3 onto
        // each of its three vertices. Virtual axis-cap vertices belong
        // to the rigid rim — their share is silently absorbed by the
        // hub (the rim does not have an independent DoF here).
        let p_ref = self.params.pressure_pa.max(0.0);
        if p_ref > 0.0 && self.cavity_volume_rest > 0.0 {
            let v_now = self.cavity_volume();
            let gamma = self.params.cavity_polytropic_gamma.max(0.1);
            let ratio = (self.cavity_volume_rest / v_now).powf(gamma);
            // Clamp the dynamic pressure to a sane band: catastrophic
            // collapse of the cavity (V → 0) would otherwise produce a
            // numerical singularity.
            // Carcass-leakage factor: in a real tire the belt/ply package
            // absorbs most of the cavity hoop stress so the bead doesn't
            // freely balloon under inflation; only a small fraction of the
            // raw cavity traction actually reaches the rubber surface as
            // bulk load. Until anisotropic belts (stage (c)) carry their
            // own share, scale the traction so the cavity gives load-
            // transfer physics without ballooning the rest geometry.
            let coupling = self.params.cavity_carcass_coupling.clamp(0.0, 1.0);
            let p_now = (p_ref * ratio * coupling).clamp(0.0, 50.0 * p_ref.max(1.0e4));
            let half_w = self.params.width * 0.5;
            let neg_axis = [self.hub[0], self.hub[1], -half_w];
            let pos_axis = [self.hub[0], self.hub[1], half_w];
            // Closure-free helper to keep the borrow checker happy.
            // For each facet vertex we add `P · (e1 × e2) / 6`, i.e.
            // `P · area · n / 3` (since `(e1 × e2)` has magnitude `2·area`).
            for tri in &self.cavity_tube_tris {
                let a = self.pos[tri[0] as usize];
                let b = self.pos[tri[1] as usize];
                let c = self.pos[tri[2] as usize];
                let e1 = sub(b, a);
                let e2 = sub(c, a);
                let n_area2 = cross(e1, e2);
                let fp = scale(n_area2, p_now / 6.0);
                force[tri[0] as usize] = add(force[tri[0] as usize], fp);
                force[tri[1] as usize] = add(force[tri[1] as usize], fp);
                force[tri[2] as usize] = add(force[tri[2] as usize], fp);
            }
            for edge in &self.cavity_neg_edges {
                let a = neg_axis;
                let b = self.pos[edge[1] as usize];
                let c = self.pos[edge[0] as usize];
                let e1 = sub(b, a);
                let e2 = sub(c, a);
                let n_area2 = cross(e1, e2);
                let fp = scale(n_area2, p_now / 6.0);
                // Axis vertex share is absorbed by the rigid rim.
                force[edge[1] as usize] = add(force[edge[1] as usize], fp);
                force[edge[0] as usize] = add(force[edge[0] as usize], fp);
            }
            for edge in &self.cavity_pos_edges {
                let a = pos_axis;
                let b = self.pos[edge[0] as usize];
                let c = self.pos[edge[1] as usize];
                let e1 = sub(b, a);
                let e2 = sub(c, a);
                let n_area2 = cross(e1, e2);
                let fp = scale(n_area2, p_now / 6.0);
                force[edge[0] as usize] = add(force[edge[0] as usize], fp);
                force[edge[1] as usize] = add(force[edge[1] as usize], fp);
            }
        }
        // Legacy direct tread-pressure (deprecated by `cavity_polytropic_gamma`).
        // Kept on a `pressure_load_scale > 0` opt-in for any caller still using it.
        if self.params.pressure_load_scale > 0.0 {
            let arc =
                self.params.outer_radius * std::f64::consts::TAU / self.params.segments as f64;
            let area_per_tread_node = arc * self.params.width / self.params.width_layers as f64
                * self.params.pressure_load_scale;
            for &id in &self.tread_nodes {
                let radial = normalize([
                    self.pos[id][0] - self.hub[0],
                    self.pos[id][1] - self.hub[1],
                    0.0,
                ]);
                force[id] = add(
                    force[id],
                    scale(radial, self.params.pressure_pa * area_per_tread_node),
                );
            }
        }

        // Note: NO explicit floor penalty force here. The contact is now a
        // hard positional projection (Lagrange-multiplier zero-gap), applied
        // after the velocity/position update below.

        // Stage (d) — Coulomb friction caps come from the runtime
        // `SurfaceMaterial`, so the same FEM tire produces different
        // grip on asphalt, dirt, sand, ice.
        let mu_static = self.params.surface_material.mu_static;
        let mu_dynamic = self.params.surface_material.mu_dynamic;
        let terrain_mu_ref = finite_positive_or(terrain.friction(), mu_static.max(1.0e-3), 1.0e-6);
        let bristle_damping = self.params.surface_material.bristle_damping;
        let cohesion = self.params.surface_material.cohesion;
        let damping = self.params.node_damping;
        let k_brush = self.params.tread_shear_stiffness * self.tread_area_per_node;
        let area_node = self.tread_area_per_node;
        let bristle_relax = self.params.bristle_relax_rate.max(0.0);

        for i in 0..self.pos.len() {
            // Implicit damping in the velocity update:
            //   m·dv/dt = F − c·v ⇒ v_new = (m·v + F·dt) / (m + c·dt)
            // This is unconditionally stable for damping at any dt.
            let m = self.mass[i];
            let semi_implicit_stiffness =
                finite_nonnegative(self.params.semi_implicit_stiffness_scale)
                    * finite_nonnegative(node_stiffness_diag[i]);
            let denom = m + damping * dt + semi_implicit_stiffness * dt * dt;
            for c in 0..3 {
                self.vel[i][c] = (m * self.vel[i][c] + force[i][c] * dt) / denom;
            }
            for c in 0..3 {
                self.vel[i][c] = self.vel[i][c].clamp(-8.0, 8.0);
            }

            // Trial position step.
            let proposed = [
                self.pos[i][0] + self.vel[i][0] * dt,
                self.pos[i][1] + self.vel[i][1] * dt,
                self.pos[i][2] + self.vel[i][2] * dt,
            ];

            let tread_slot = self.node_to_tread[i].map(|t| t as usize);

            let tread_contact_radius = if tread_slot.is_some() {
                (area_node / std::f64::consts::PI)
                    .sqrt()
                    .min(self.params.width * 0.35)
                    .max(0.0)
            } else {
                0.0
            };
            let manifold = if tread_slot.is_some() {
                terrain.contact_manifold(proposed, tread_contact_radius, 1.0e-4, 6)
            } else {
                Vec::new()
            };
            let (signed_gap, contact_normal, manifold_mu) = if manifold.is_empty() {
                let (gap, raw_contact_normal) = terrain.signed_distance_and_normal(proposed);
                (
                    gap,
                    normalize(raw_contact_normal),
                    finite_nonnegative(terrain.friction_at(proposed[0], proposed[2])),
                )
            } else {
                let mut normal_acc = [0.0; 3];
                let mut mu_acc = 0.0;
                let mut w_acc = 0.0;
                let mut deepest_pen = 0.0_f64;
                for patch in &manifold {
                    let w = patch.penetration.max(1.0e-6);
                    normal_acc = add(normal_acc, scale(patch.normal, w));
                    mu_acc += patch.friction * w;
                    w_acc += w;
                    deepest_pen = deepest_pen.max(patch.penetration);
                }
                (
                    -deepest_pen,
                    normalize(normal_acc),
                    finite_nonnegative(mu_acc / w_acc.max(1.0e-12)),
                )
            };
            if signed_gap < 0.0 {
                // ── XPBD zero-gap contact + brush/Coulomb friction ──
                //
                // Project along the local terrain normal; the projection
                // magnitude (`pen`) stands in for the accumulated normal
                // impulse λ_n in the XPBD friction recipe
                // `|Δx_t| ≤ μ · |Δx_n|`.
                let pen = -signed_gap;
                self.pos[i] = add(proposed, scale(contact_normal, pen));
                let ground_v = terrain.surface_velocity(proposed[0], proposed[2]);
                let terrain_mu = manifold_mu;
                let friction_scale = (terrain_mu / terrain_mu_ref).clamp(0.0, 3.0);
                let mu_static_local = mu_static * friction_scale;
                let mu_dynamic_local = mu_dynamic * friction_scale;

                // PLAN_TOURQUE Stage C — spin contribution to the tread
                // node's world velocity at the contact patch.
                //
                // Sign convention: `hub_omega > 0` ↔ forward rolling
                // (top tread moves +X, bottom moves −X). Using r =
                // pos − hub, the spin velocity is v_spin = [ω·r_y,
                //   0, −ω·r_x] for our spin-axis-along-Z convention.
                // Only tread nodes get this contribution because only
                // they are coupled to the brush model and only their
                // bristles drive the longitudinal friction.
                let (spin_v_x, spin_v_z) = if tread_slot.is_some() {
                    let r_x = proposed[0] - self.hub[0];
                    let r_y = proposed[1] - self.hub[1];
                    (self.hub_omega * r_y, -self.hub_omega * r_x)
                } else {
                    (0.0, 0.0)
                };

                // Trial tangential motion (positional Coulomb cone).
                let mut tx = (self.vel[i][0] + spin_v_x - ground_v[0]) * dt;
                let mut tz = (self.vel[i][2] + spin_v_z - ground_v[2]) * dt;

                // Stage (d) — bristle/brush term for tread nodes. The
                // brush traction is `K·A·bristle + c_b·A·v_tangent`,
                // capped by the same cone as the Coulomb projection.
                // We express it as an *extra positional correction*
                // applied to the trial motion, so a single cone-clip
                // below caps everything consistently.
                if let Some(ti) = tread_slot {
                    let bristle = self.tread_bristle[ti];
                    // Brush + viscous damping → an effective tangential
                    // impulse that we resolve as a position offset. The
                    // viscous term sees `vel + spin − ground` so a
                    // spinning hub with stationary nodes still produces
                    // tread slip.
                    let brush_fx = k_brush * bristle[0]
                        + bristle_damping * area_node * (self.vel[i][0] + spin_v_x - ground_v[0]);
                    let brush_fz = k_brush * bristle[2]
                        + bristle_damping * area_node * (self.vel[i][2] + spin_v_z - ground_v[2]);
                    let bx = brush_fx * dt * dt / m.max(1.0e-9);
                    let bz = brush_fz * dt * dt / m.max(1.0e-9);
                    tx -= bx;
                    tz -= bz;
                    // PLAN_TOURQUE Stage D — accumulate the brush
                    // reaction. The bristle implements Newton's third
                    // law against the ground: the per-node brush_f is
                    // the force the *ground* applies to the tread, so
                    // summed across contact nodes it is the net
                    // longitudinal/lateral wrench the contact patch
                    // transmits back to the hub.
                    net_contact_fx -= brush_fx;
                    net_contact_fz -= brush_fz;
                }

                let t_mag = (tx * tx + tz * tz).sqrt();
                let cohesion_cap = cohesion * area_node;
                let cap_static = mu_static_local * pen + cohesion_cap;
                if t_mag <= cap_static {
                    // Stick.
                    self.pos[i][0] += ground_v[0] * dt;
                    self.pos[i][2] += ground_v[2] * dt;
                    self.vel[i][0] = ground_v[0];
                    self.vel[i][2] = ground_v[2];
                } else if t_mag > 1.0e-12 {
                    // Slide.
                    let keep = (t_mag - mu_dynamic_local * pen - cohesion_cap) / t_mag;
                    let keep = keep.max(0.0);
                    self.pos[i][0] += ground_v[0] * dt + tx * keep;
                    self.pos[i][2] += ground_v[2] * dt + tz * keep;
                    self.vel[i][0] = ground_v[0] + (self.vel[i][0] - ground_v[0]) * keep;
                    self.vel[i][2] = ground_v[2] + (self.vel[i][2] - ground_v[2]) * keep;
                }
                if self.vel[i][1] < 0.0 {
                    self.vel[i][1] = 0.0;
                }

                // Integrate the bristle for tread nodes that are in
                // contact this substep. PLAN_TOURQUE Stage C: the
                // tangential slip includes the spin contribution so the
                // bristle stretches under a spinning hub even when the
                // FEM node itself is momentarily at rest.
                if let Some(ti) = tread_slot {
                    let v_tan = [
                        self.vel[i][0] + spin_v_x - ground_v[0],
                        0.0,
                        self.vel[i][2] + spin_v_z - ground_v[2],
                    ];
                    let b = &mut self.tread_bristle[ti];
                    b[0] += v_tan[0] * dt;
                    b[2] += v_tan[2] * dt;
                    // Saturate the bristle at the Coulomb-cone limit so
                    // |F_brush| stays consistent with the slide branch.
                    let cap =
                        (mu_dynamic_local * pen / dt.max(1.0e-9)).max(0.0) / k_brush.max(1.0e-9);
                    let mag = (b[0] * b[0] + b[2] * b[2]).sqrt();
                    if mag > cap && mag > 1.0e-12 {
                        let s = cap / mag;
                        b[0] *= s;
                        b[2] *= s;
                    }
                }
            } else {
                self.pos[i] = proposed;
                // Tread node airborne → bristle relaxes back to zero.
                if let Some(ti) = tread_slot {
                    let decay = (-bristle_relax * dt).exp();
                    let b = &mut self.tread_bristle[ti];
                    b[0] *= decay;
                    b[2] *= decay;
                }
            }
        }

        // Stage (f) — XPBD near-incompressibility/volume preservation.
        // Elasticity remains explicit, but this constraint pass keeps the
        // volumetric tire from silently crushing tetrahedra under heavy load
        // before the full implicit FEM/Newton solve exists.
        self.project_structural_constraints(dt, terrain);
        self.project_volume_constraints(dt, terrain);

        // Hub — implicit damping for consistency.
        let hub_damp = 260.0_f64;
        let hub_denom = self.params.hub_mass + hub_damp * dt;
        self.hub_v_y = (self.params.hub_mass * self.hub_v_y + hub_force_y * dt) / hub_denom;
        self.hub_v_y = self.hub_v_y.clamp(-2.5, 2.5);
        self.hub[1] += self.hub_v_y * dt;
        let hub_floor = self.params.inner_radius * 0.75;
        if self.hub[1] < hub_floor {
            self.hub[1] = hub_floor;
            if self.hub_v_y < 0.0 {
                self.hub_v_y = 0.0;
            }
        }
        // PLAN_TOURQUE Stage D — when the hub is dynamic, integrate
        // `hub_v_xz` from the contact reaction accumulated above and
        // advance the hub XZ position. Same implicit-damping shape as
        // the vertical hub integrator so the integration timescale is
        // consistent. `hub_kinematic = true` (default) leaves
        // `hub_v_xz` untouched so existing scenes — including scene
        // 122's rolling-road bench — keep their current behaviour.
        if !self.hub_kinematic {
            let hub_mass = self.params.hub_mass.max(1.0e-6);
            let hub_drag_xz = 80.0_f64;
            let xz_denom = hub_mass + hub_drag_xz * dt;
            self.hub_v_xz[0] = (hub_mass * self.hub_v_xz[0] + net_contact_fx * dt) / xz_denom;
            self.hub_v_xz[1] = (hub_mass * self.hub_v_xz[1] + net_contact_fz * dt) / xz_denom;
            // Bound the hub speed to the same clamp the brush model
            // uses for the per-node velocity (8 m/s), so a runaway
            // integration cannot bypass that limit.
            for axis in 0..2 {
                if !self.hub_v_xz[axis].is_finite() {
                    self.hub_v_xz[axis] = 0.0;
                }
                self.hub_v_xz[axis] = self.hub_v_xz[axis].clamp(-8.0, 8.0);
            }
            self.hub[0] += self.hub_v_xz[0] * dt;
            self.hub[2] += self.hub_v_xz[1] * dt;
        }
        // PLAN_TOURQUE Stage D — always publish the net contact reaction
        // so a future chassis can read it via `last_hub_reaction_xyz`,
        // regardless of whether the hub is kinematic or dynamic.
        self.hub_reaction_xyz = [net_contact_fx, hub_force_y, net_contact_fz];

        // PLAN_REALTIRE Stream B Stage C.6 — return the scratch buffers
        // to `self` so the next substep can reuse the allocation.
        self.scratch_force = force;
        self.scratch_node_stiffness_diag = node_stiffness_diag;
    }
}
