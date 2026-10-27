// Volumetric tire tetra-FEM step.
//
// One invocation owns one node. The kernel now uses a node→tet CSR
// adjacency (stage (h)) instead of looping over all tets, plus the
// anisotropic Holzapfel belt/ply stress (stage (c)) on top of the
// compressible Neo-Hookean carcass. It mirrors the CPU
// `molla_vehicle::TetraTireFem::step_inner`.

struct Uniforms {
    // x=node_count, y=tet_count, z=bead_count, w=tread_count
    counts: vec4<u32>,
    // x=structural_edge_count, y=sidewall_dihedral_count, zw=reserved
    counts2: vec4<u32>,
    // x=dt, y=young_modulus, z=poisson_ratio, w=pressure_pa
    phys0: vec4<f32>,
    // x=node_damping, y=bead_stiffness, z=floor_stiffness, w=floor_damping
    phys1: vec4<f32>,
    // x=pressure_area_per_tread_node, yzw=hub
    phys2: vec4<f32>,
    // x=gravity_y, y=cavity_radial_force_per_bead, z=belt_proxy_mu,
    // w=xpbd_volume_compliance
    //
    // `cavity_radial_force_per_bead` is `P_now · carcass_leak · A_per_bead`
    // pre-computed CPU-side from the divergence cavity-volume integral.
    // `belt_proxy_mu` is added uniformly to every tet's μ — it lives in
    // the uniforms (not the per-tet buffer) so the persistent path
    // doesn't have to re-upload `tet_mu` when pressure changes.
    phys3: vec4<f32>,
    // x=xpbd_bead_compliance, y=semi_implicit_stiffness_scale,
    // z=sidewall_ply_stretch_limit, w=sidewall_ply_compliance
    phys4: vec4<f32>,
    // PLAN_TOURQUE Stage G — torque/spin extension.
    //
    //   x = hub_omega (rad/s, signed; +ω = forward rolling)
    //   y = drive_torque_n_m (N·m, signed; +tau spins +ω)
    //   z = brake_torque_n_m (N·m, ≥0; always opposes ω)
    //   w = reserved (future: spin_damping coefficient)
    //
    // Stage G part 1 wires these into the uniforms layout so the host
    // can already upload values; the per-node physics that consumes
    // them (spin-coupled bristle slip + dynamic-hub XZ integration)
    // is the open Stage G part 2 follow-up tracked in
    // `TireFemTorqueGpuContract.status`.
    phys5: vec4<f32>,
};

@group(0) @binding(0) var<storage, read>       rest_pos:     array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       pos_in:       array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       vel_in:       array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       mass_in:      array<f32>;
@group(0) @binding(4) var<storage, read>       tet_indices:  array<u32>;
@group(0) @binding(5) var<storage, read>       tet_grads:    array<vec4<f32>>;
@group(0) @binding(6) var<storage, read>       tet_volume:   array<f32>;
@group(0) @binding(7) var<storage, read>       tet_mu:       array<f32>;
@group(0) @binding(8) var<storage, read>       tet_lambda:   array<f32>;
@group(0) @binding(9) var<storage, read>       bead_ids:     array<u32>;
@group(0) @binding(10) var<storage, read>      bead_offsets: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read>      tread_ids:    array<u32>;

@group(0) @binding(12) var<storage, read_write> pos_out:   array<vec4<f32>>;
@group(0) @binding(13) var<storage, read_write> vel_out:   array<vec4<f32>>;
@group(0) @binding(14) var<storage, read_write> force_out: array<vec4<f32>>;

// Stage (c) — anisotropic Holzapfel fibre data (per-tet).
@group(0) @binding(15) var<storage, read>       tet_fiber_dir: array<vec4<f32>>;
@group(0) @binding(16) var<storage, read>       tet_fiber_k1:  array<f32>;
@group(0) @binding(17) var<storage, read>       tet_fiber_k2:  array<f32>;

// Stage (h) CSR — `node_tet_entries[node_tet_offsets[i]..[i+1]]`
// packs `(tet_index << 2) | local_vertex` for every tet incident on node i.
@group(0) @binding(18) var<storage, read>       node_tet_offsets: array<u32>;
@group(0) @binding(19) var<storage, read>       node_tet_entries: array<u32>;

// Stage (f) — hub state lives in storage so the GPU hub reduction
// kernel can write it without a uniform round-trip.
//   hub_state[0] = (hub_x, hub_y, hub_z, hub_v_y)
@group(0) @binding(20) var<storage, read>       hub_state: array<vec4<f32>>;

// Stage (c2) — second explicit crossed ply family.
@group(0) @binding(21) var<storage, read>       tet_fiber_dir_b: array<vec4<f32>>;
@group(0) @binding(22) var<storage, read>       tet_fiber_k1_b:  array<f32>;
@group(0) @binding(23) var<storage, read>       tet_fiber_k2_b:  array<f32>;

// Stage (f) — XPBD structural edge graph. `xpbd_edge_indices[e].xy`
// are endpoint node ids; `xpbd_edge_params[e].x` is rest length and
// `.y` is compliance. Node-edge CSR entries pack `(edge_index << 1) | local`.
@group(0) @binding(24) var<storage, read>       xpbd_edge_indices: array<vec4<u32>>;
@group(0) @binding(25) var<storage, read>       xpbd_edge_params:  array<vec4<f32>>;
@group(0) @binding(26) var<storage, read>       node_edge_offsets: array<u32>;
@group(0) @binding(27) var<storage, read>       node_edge_entries: array<u32>;

// Stage (c3) — third/fourth explicit ply families for cap/hoop/radial
// multi-ply stacks.
@group(0) @binding(28) var<storage, read>       tet_fiber_dir_c: array<vec4<f32>>;
@group(0) @binding(29) var<storage, read>       tet_fiber_k1_c:  array<f32>;
@group(0) @binding(30) var<storage, read>       tet_fiber_k2_c:  array<f32>;
@group(0) @binding(31) var<storage, read>       tet_fiber_dir_d: array<vec4<f32>>;
@group(0) @binding(32) var<storage, read>       tet_fiber_k1_d:  array<f32>;
@group(0) @binding(33) var<storage, read>       tet_fiber_k2_d:  array<f32>;

// Stage (f3) — XPBD sidewall dihedral-angle constraints. Entries pack
// `(dihedral_index << 2) | local_vertex`.
@group(0) @binding(34) var<storage, read>       xpbd_dihedral_indices: array<vec4<u32>>;
@group(0) @binding(35) var<storage, read>       xpbd_dihedral_params:  array<vec4<f32>>;
@group(0) @binding(36) var<storage, read>       node_dihedral_offsets: array<u32>;
@group(0) @binding(37) var<storage, read>       node_dihedral_entries: array<u32>;

@group(1) @binding(0) var<uniform> u: Uniforms;

fn safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let l2 = dot(v, v);
    if (l2 > 1.0e-14) {
        return v * inverseSqrt(l2);
    }
    return fallback;
}

fn det3(m: mat3x3<f32>) -> f32 {
    return dot(m[0], cross(m[1], m[2]));
}

fn tetra_deformation(ib: u32) -> mat3x3<f32> {
    let origin = pos_in[tet_indices[ib]].xyz;
    var f0 = vec3<f32>(0.0);
    var f1 = vec3<f32>(0.0);
    var f2 = vec3<f32>(0.0);
    for (var a: u32 = 1u; a < 4u; a = a + 1u) {
        let x = pos_in[tet_indices[ib + a]].xyz - origin;
        let g = tet_grads[ib + a].xyz;
        f0 = f0 + x.x * g;
        f1 = f1 + x.y * g;
        f2 = f2 + x.z * g;
    }
    return transpose(mat3x3<f32>(f0, f1, f2));
}

fn tetra_gradient(ib: u32, local: u32) -> vec3<f32> {
    if (local == 0u) {
        return -(tet_grads[ib + 1u].xyz + tet_grads[ib + 2u].xyz + tet_grads[ib + 3u].xyz);
    }
    return tet_grads[ib + local].xyz;
}

fn inverse_transpose3(m: mat3x3<f32>) -> mat3x3<f32> {
    let c0 = cross(m[1], m[2]);
    let c1 = cross(m[2], m[0]);
    let c2 = cross(m[0], m[1]);
    let d = max(abs(dot(m[0], c0)), 1.0e-8) * sign(dot(m[0], c0));
    return mat3x3<f32>(c0 / d, c1 / d, c2 / d);
}


fn tetra_signed_volume(x0: vec3<f32>, x1: vec3<f32>, x2: vec3<f32>, x3: vec3<f32>) -> f32 {
    return det3(mat3x3<f32>(x1 - x0, x2 - x0, x3 - x0)) / 6.0;
}

fn tetra_volume_grad(local: u32, x0: vec3<f32>, x1: vec3<f32>, x2: vec3<f32>, x3: vec3<f32>) -> vec3<f32> {
    if (local == 0u) {
        return cross(x3 - x1, x2 - x1) / 6.0;
    }
    if (local == 1u) {
        return cross(x2 - x0, x3 - x0) / 6.0;
    }
    if (local == 2u) {
        return cross(x3 - x0, x1 - x0) / 6.0;
    }
    return cross(x1 - x0, x2 - x0) / 6.0;
}


@compute @workgroup_size(64)
fn step_nodes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.counts.x) { return; }

    let p = pos_in[i].xyz;
    let v = vel_in[i].xyz;
    let m = mass_in[i];
    if (m <= 0.0) {
        pos_out[i] = vec4<f32>(p, 0.0);
        vel_out[i] = vec4<f32>(0.0);
        force_out[i] = vec4<f32>(0.0);
        return;
    }

    // Stage-(a)/(e) integrator: damping is folded into the implicit
    // velocity update below; only gravity goes into `force` here.
    var force = vec3<f32>(0.0, m * u.phys3.x, 0.0);
    var node_stiffness_diag: f32 = 0.0;

    // Tetra hyperelastic forces. Stage (h) CSR: iterate only over the
    // tets incident on this node, packed as
    //   entry = (tet_index << 2) | local_vertex_id
    let csr_lo = node_tet_offsets[i];
    let csr_hi = node_tet_offsets[i + 1u];
    for (var e: u32 = csr_lo; e < csr_hi; e = e + 1u) {
        let entry = node_tet_entries[e];
        let t = entry >> 2u;
        let local = entry & 3u;
        let ib = t * 4u;

        let F = tetra_deformation(ib);
        let FinvT = inverse_transpose3(F);
        let J = max(det3(F), 0.20);
        let logJ = log(J);
        // Stage (b)/(c) — uniform belt-proxy μ added on top of per-tet μ.
        let mu = tet_mu[t] + u.phys3.z;
        let lambda = tet_lambda[t];
        var P = mu * (F - FinvT) + lambda * logJ * FinvT;

        // Stage (c/c2) — crossed Holzapfel ply families.
        let g_node = tetra_gradient(ib, local);
        var fibre_tangent = 0.0;
        let fiber0 = add_fiber_stress(P, F, tet_fiber_dir[t].xyz, tet_fiber_k1[t], tet_fiber_k2[t], g_node);
        P = fiber0.stress;
        fibre_tangent = fibre_tangent + fiber0.tangent;
        let fiber1 = add_fiber_stress(P, F, tet_fiber_dir_b[t].xyz, tet_fiber_k1_b[t], tet_fiber_k2_b[t], g_node);
        P = fiber1.stress;
        fibre_tangent = fibre_tangent + fiber1.tangent;
        let fiber2 = add_fiber_stress(P, F, tet_fiber_dir_c[t].xyz, tet_fiber_k1_c[t], tet_fiber_k2_c[t], g_node);
        P = fiber2.stress;
        fibre_tangent = fibre_tangent + fiber2.tangent;
        let fiber3 = add_fiber_stress(P, F, tet_fiber_dir_d[t].xyz, tet_fiber_k1_d[t], tet_fiber_k2_d[t], g_node);
        P = fiber3.stress;
        fibre_tangent = fibre_tangent + fiber3.tangent;
        let sg = P * g_node;
        force = force - tet_volume[t] * sg;

        // Stage (f2) — CPU/GPU parity for the diagonal semi-implicit
        // stiffness stabilizer. This is a lumped tangent estimate, not a
        // full Newton solve; it mirrors the CPU denominator
        // `m + c·dt + scale·k_diag·dt²`.
        let grad_norm2 = dot(g_node, g_node);
        node_stiffness_diag = node_stiffness_diag
            + tet_volume[t] * ((abs(mu) + abs(lambda)) * grad_norm2 + fibre_tangent);
    }

    // Bead/rim coupling to the current hub pose + cavity-pressure
    // radial outward traction on every bead node. The cavity gas-law
    // factor `phys3.y` is `P_now · carcass_leak · A_per_bead`, which the
    // CPU computes from `cavity_volume()`. The hub is read from the
    // storage `hub_state` buffer (stage (f)) so GPU-resident hub
    // dynamics close the loop.
    let hub = hub_state[0].xyz;
    let cavity_per_bead = u.phys3.y;
    for (var b: u32 = 0u; b < u.counts.z; b = b + 1u) {
        if (bead_ids[b] == i) {
            let bead_target = hub + bead_offsets[b].xyz;
            force = force + u.phys1.y * (bead_target - p);
            node_stiffness_diag = node_stiffness_diag + max(u.phys1.y, 0.0);
            if (cavity_per_bead > 0.0) {
                let radial = safe_normalize(
                    vec3<f32>(p.x - hub.x, p.y - hub.y, 0.0),
                    vec3<f32>(1.0, 0.0, 0.0),
                );
                force = force + radial * cavity_per_bead;
            }
        }
    }

    // Legacy tread-pressure traction (kept for opt-in callers that set
    // `pressure_load_scale > 0`; default `phys2.x = 0` makes it a no-op).
    for (var q: u32 = 0u; q < u.counts.w; q = q + 1u) {
        if (tread_ids[q] == i) {
            let radial = safe_normalize(vec3<f32>(p.x - hub.x, p.y - hub.y, 0.0), vec3<f32>(0.0, 1.0, 0.0));
            force = force + radial * (u.phys0.w * u.phys2.x);
        }
    }

    // ── Stage-(a)/(e) update ─────────────────────────────────────────
    // No explicit floor penalty — replaced by a hard XPBD-style positional
    // projection below. Implicit damping in the velocity update.
    let dt = u.phys0.x;
    let damping = u.phys1.x;
    let semi_implicit_stiffness = max(u.phys4.y, 0.0) * max(node_stiffness_diag, 0.0);
    let denom = m + damping * dt + semi_implicit_stiffness * dt * dt;
    var v_next = (m * v + force * dt) / denom;
    v_next = clamp(v_next, vec3<f32>(-8.0), vec3<f32>(8.0));
    var p_next = p + v_next * dt;

    // Stage (f) — first GPU XPBD volume preservation. This mirrors the CPU
    // tet-volume projection in a race-free gather form: the node owner loops
    // over its incident tets and applies only its local correction. It is a
    // single sweep per kernel dispatch; persistent stepping performs more
    // sweeps by dispatching more substeps.
    let xpbd_volume_compliance = u.phys3.w;
    if (xpbd_volume_compliance <= 1.0e20) {
        let alpha = max(xpbd_volume_compliance, 0.0) / max(dt * dt, 1.0e-12);
        var dp_sum = vec3<f32>(0.0);
        var dp_count = 0.0;
        for (var e2: u32 = csr_lo; e2 < csr_hi; e2 = e2 + 1u) {
            let entry = node_tet_entries[e2];
            let t = entry >> 2u;
            let local = entry & 3u;
            let ib = t * 4u;
            let id0 = tet_indices[ib + 0u];
            let id1 = tet_indices[ib + 1u];
            let id2 = tet_indices[ib + 2u];
            let id3 = tet_indices[ib + 3u];
            let x0 = pos_in[id0].xyz;
            let x1 = pos_in[id1].xyz;
            let x2 = pos_in[id2].xyz;
            let x3 = pos_in[id3].xyz;
            let volume = tetra_signed_volume(x0, x1, x2, x3);
            let c = volume - tet_volume[t];
            if (abs(c) <= max(tet_volume[t], 1.0e-12) * 1.0e-4) {
                continue;
            }
            let g0 = tetra_volume_grad(0u, x0, x1, x2, x3);
            let g1 = tetra_volume_grad(1u, x0, x1, x2, x3);
            let g2 = tetra_volume_grad(2u, x0, x1, x2, x3);
            let g3 = tetra_volume_grad(3u, x0, x1, x2, x3);
            let w0 = 1.0 / max(mass_in[id0], 1.0e-9);
            let w1 = 1.0 / max(mass_in[id1], 1.0e-9);
            let w2 = 1.0 / max(mass_in[id2], 1.0e-9);
            let w3 = 1.0 / max(mass_in[id3], 1.0e-9);
            let denom_v = alpha
                + w0 * dot(g0, g0)
                + w1 * dot(g1, g1)
                + w2 * dot(g2, g2)
                + w3 * dot(g3, g3);
            if (denom_v <= 1.0e-18) {
                continue;
            }
            let lambda_v = -c / denom_v;
            var g_local = g0;
            if (local == 1u) { g_local = g1; }
            if (local == 2u) { g_local = g2; }
            if (local == 3u) { g_local = g3; }
            dp_sum = dp_sum + g_local * ((1.0 / max(m, 1.0e-9)) * lambda_v);
            dp_count = dp_count + 1.0;
        }
        if (dp_count > 0.0) {
            let dp = dp_sum / dp_count;
            p_next = p_next + dp;
            v_next = clamp(v_next + dp / max(dt, 1.0e-9), vec3<f32>(-12.0), vec3<f32>(12.0));
        }
    }

    // Stage (f) — first GPU XPBD bead/rim mount projection. This uses the
    // existing bead id/offset buffers, so it does not need float atomics or
    // a separate reduction. It complements (does not replace) the bead spring
    // force above by keeping bead nodes geometrically attached to the mounted
    // rim offsets.
    let xpbd_bead_compliance = u.phys4.x;
    if (xpbd_bead_compliance <= 1.0e20) {
        let alpha_b = max(xpbd_bead_compliance, 0.0) / max(dt * dt, 1.0e-12);
        for (var b2: u32 = 0u; b2 < u.counts.z; b2 = b2 + 1u) {
            if (bead_ids[b2] == i) {
                let bead_mount = hub + bead_offsets[b2].xyz;
                let delta = p_next - bead_mount;
                let inv_m = 1.0 / max(m, 1.0e-9);
                let denom_b = inv_m + alpha_b;
                if (denom_b > 1.0e-18) {
                    let dp_b = delta * (-inv_m / denom_b);
                    p_next = p_next + dp_b;
                    v_next = clamp(v_next + dp_b / max(dt, 1.0e-9), vec3<f32>(-12.0), vec3<f32>(12.0));
                }
            }
        }
    }

    // Stage (f) — first GPU structural XPBD edge pass for tread belt and
    // sidewall bend-proxy constraints. Race-free gather form: each node
    // applies only its endpoint correction from incident structural edges.
    if (u.counts2.x > 0u) {
        let edge_lo = node_edge_offsets[i];
        let edge_hi = node_edge_offsets[i + 1u];
        var dp_edge_sum = vec3<f32>(0.0);
        var dp_edge_count = 0.0;
        for (var ee: u32 = edge_lo; ee < edge_hi; ee = ee + 1u) {
            let packed = node_edge_entries[ee];
            let eidx = packed >> 1u;
            if (eidx >= u.counts2.x) {
                continue;
            }
            let local = packed & 1u;
            let edge = xpbd_edge_indices[eidx];
            let a_id = edge.x;
            let b_id = edge.y;
            let xa = pos_in[a_id].xyz;
            let xb = pos_in[b_id].xyz;
            let dx = xb - xa;
            let len = length(dx);
            let rest_len = xpbd_edge_params[eidx].x;
            if (len <= 1.0e-9 || rest_len <= 1.0e-9) {
                continue;
            }
            let c_edge = len - rest_len;
            if (abs(c_edge) <= max(rest_len, 1.0e-9) * 1.0e-5) {
                continue;
            }
            let n_edge = dx / len;
            let wa = 1.0 / max(mass_in[a_id], 1.0e-9);
            let wb = 1.0 / max(mass_in[b_id], 1.0e-9);
            let compliance = xpbd_edge_params[eidx].y;
            let alpha_e = max(compliance, 0.0) / max(dt * dt, 1.0e-12);
            let denom_e = wa + wb + alpha_e;
            if (denom_e <= 1.0e-18) {
                continue;
            }
            let lambda_e = -c_edge / denom_e;
            if (local == 0u) {
                dp_edge_sum = dp_edge_sum + n_edge * (-wa * lambda_e);
            } else {
                dp_edge_sum = dp_edge_sum + n_edge * (wb * lambda_e);
            }
            dp_edge_count = dp_edge_count + 1.0;
        }
        if (dp_edge_count > 0.0) {
            let dp_edge = dp_edge_sum / dp_edge_count;
            p_next = p_next + dp_edge;
            v_next = clamp(v_next + dp_edge / max(dt, 1.0e-9), vec3<f32>(-12.0), vec3<f32>(12.0));
        }
    }

    // Stage (f3) — true sidewall quad dihedral-angle XPBD projection.
    // Like the CPU implementation, this preserves the fold angle between
    // triangles (a,b,c) and (a,c,d). The gather form applies only this
    // node's local correction and avoids atomics; multiple persistent
    // substeps provide additional sweeps.
    if (u.counts2.y > 0u) {
        let diag_lo = node_dihedral_offsets[i];
        let diag_hi = node_dihedral_offsets[i + 1u];
        var dp_diag_sum = vec3<f32>(0.0);
        var dp_diag_count = 0.0;
        for (var dd: u32 = diag_lo; dd < diag_hi; dd = dd + 1u) {
            let packed = node_dihedral_entries[dd];
            let didx = packed >> 2u;
            if (didx >= u.counts2.y) {
                continue;
            }
            let local = packed & 3u;
            let ids = xpbd_dihedral_indices[didx];
            let x0 = pos_in[ids.x].xyz;
            let x1 = pos_in[ids.y].xyz;
            let x2 = pos_in[ids.z].xyz;
            let x3 = pos_in[ids.w].xyz;
            let rest_angle = xpbd_dihedral_params[didx].x;
            let c_diag = quad_dihedral_angle(x0, x1, x2, x3) - rest_angle;
            if (abs(c_diag) <= 1.0e-5) {
                continue;
            }
            let gradients = quad_dihedral_gradients(x0, x1, x2, x3);
            let g0 = gradients[0];
            let g1 = gradients[1];
            let g2 = gradients[2];
            let g3 = gradients[3];
            let w0 = 1.0 / max(mass_in[ids.x], 1.0e-9);
            let w1 = 1.0 / max(mass_in[ids.y], 1.0e-9);
            let w2 = 1.0 / max(mass_in[ids.z], 1.0e-9);
            let w3 = 1.0 / max(mass_in[ids.w], 1.0e-9);
            let compliance = xpbd_dihedral_params[didx].y;
            let alpha_d = max(compliance, 0.0) / max(dt * dt, 1.0e-12);
            let denom_d = alpha_d
                + w0 * dot(g0, g0)
                + w1 * dot(g1, g1)
                + w2 * dot(g2, g2)
                + w3 * dot(g3, g3);
            if (denom_d <= 1.0e-18) {
                continue;
            }
            let lambda_d = -c_diag / denom_d;
            var g_local = g0;
            var w_local = w0;
            if (local == 1u) { g_local = g1; w_local = w1; }
            if (local == 2u) { g_local = g2; w_local = w2; }
            if (local == 3u) { g_local = g3; w_local = w3; }
            dp_diag_sum = dp_diag_sum + g_local * (w_local * lambda_d);
            dp_diag_count = dp_diag_count + 1.0;
        }
        if (dp_diag_count > 0.0) {
            let dp_diag = dp_diag_sum / dp_diag_count;
            p_next = p_next + dp_diag;
            v_next = clamp(v_next + dp_diag / max(dt, 1.0e-9), vec3<f32>(-12.0), vec3<f32>(12.0));
        }
    }

    // Stage (f4) — tensile sidewall ply-strain XPBD guardrail. This is the
    // GPU counterpart to the CPU `project_sidewall_ply_strain_constraints`
    // step: sidewall tets are identified by radial + axial helper plies, then
    // every explicit sidewall ply family is constrained by |F*a0| <= limit.
    // It is still a compact ply-stretch guardrail, not a full laminated shell.
    let ply_limit = max(u.phys4.z, 1.01);
    let ply_compliance = u.phys4.w;
    if (ply_compliance <= 1.0e20 && ply_limit > 1.0) {
        let alpha_p = max(ply_compliance, 0.0) / max(dt * dt, 1.0e-12);
        var dp_ply_sum = vec3<f32>(0.0);
        var dp_ply_count = 0.0;
        for (var pe: u32 = csr_lo; pe < csr_hi; pe = pe + 1u) {
            let entry = node_tet_entries[pe];
            let t = entry >> 2u;
            let local = entry & 3u;
            let ib = t * 4u;
            if (tet_fiber_k1[t] <= 0.0 || tet_fiber_k1_b[t] <= 0.0 || tet_fiber_k1_c[t] <= 0.0 || tet_fiber_k1_d[t] <= 0.0) {
                continue;
            }
            let id0 = tet_indices[ib + 0u];
            let id1 = tet_indices[ib + 1u];
            let id2 = tet_indices[ib + 2u];
            let id3 = tet_indices[ib + 3u];
            let r0 = rest_pos[id0].xyz;
            let r1 = rest_pos[id1].xyz;
            let r2 = rest_pos[id2].xyz;
            let r3 = rest_pos[id3].xyz;
            let centroid = (r0 + r1 + r2 + r3) * 0.25;
            let radial = safe_normalize(vec3<f32>(centroid.x - hub.x, centroid.y - hub.y, 0.0), vec3<f32>(1.0, 0.0, 0.0));
            let axial = vec3<f32>(0.0, 0.0, 1.0);
            if (abs(dot(tet_fiber_dir_c[t].xyz, radial)) < 0.85 || abs(dot(tet_fiber_dir_d[t].xyz, axial)) < 0.85) {
                continue;
            }
            let Fp = tetra_deformation(ib);
            for (var family: u32 = 0u; family < 4u; family = family + 1u) {
                var dir = tet_fiber_dir[t].xyz;
                if (family == 1u) { dir = tet_fiber_dir_b[t].xyz; }
                if (family == 2u) { dir = tet_fiber_dir_c[t].xyz; }
                if (family == 3u) { dir = tet_fiber_dir_d[t].xyz; }
                let fa = Fp * dir;
                let stretch = length(fa);
                let c_ply = stretch - ply_limit;
                if (stretch <= 1.0e-9 || c_ply <= 1.0e-6) {
                    continue;
                }
                let stretch_grad_dir = fa / stretch;
                let g0s = dot(tetra_gradient(ib, 0u), dir);
                let g1s = dot(tet_grads[ib + 1u].xyz, dir);
                let g2s = dot(tet_grads[ib + 2u].xyz, dir);
                let g3s = dot(tet_grads[ib + 3u].xyz, dir);
                let pg0 = stretch_grad_dir * g0s;
                let pg1 = stretch_grad_dir * g1s;
                let pg2 = stretch_grad_dir * g2s;
                let pg3 = stretch_grad_dir * g3s;
                let w0 = 1.0 / max(mass_in[id0], 1.0e-9);
                let w1 = 1.0 / max(mass_in[id1], 1.0e-9);
                let w2 = 1.0 / max(mass_in[id2], 1.0e-9);
                let w3 = 1.0 / max(mass_in[id3], 1.0e-9);
                let denom_p = alpha_p
                    + w0 * dot(pg0, pg0)
                    + w1 * dot(pg1, pg1)
                    + w2 * dot(pg2, pg2)
                    + w3 * dot(pg3, pg3);
                if (denom_p <= 1.0e-18) {
                    continue;
                }
                let lambda_p = -c_ply / denom_p;
                var g_local_p = pg0;
                var w_local_p = w0;
                if (local == 1u) { g_local_p = pg1; w_local_p = w1; }
                if (local == 2u) { g_local_p = pg2; w_local_p = w2; }
                if (local == 3u) { g_local_p = pg3; w_local_p = w3; }
                dp_ply_sum = dp_ply_sum + g_local_p * (w_local_p * lambda_p);
                dp_ply_count = dp_ply_count + 1.0;
            }
        }
        if (dp_ply_count > 0.0) {
            let dp_ply = dp_ply_sum / dp_ply_count;
            p_next = p_next + dp_ply;
            v_next = clamp(v_next + dp_ply / max(dt, 1.0e-9), vec3<f32>(-12.0), vec3<f32>(12.0));
        }
    }

    // XPBD zero-gap contact + Coulomb friction.
    //
    // PLAN_TOURQUE Stage G part 2 — spin contribution to the tangential
    // slip. Sign convention matches the CPU `step_inner`:
    //   `hub_omega > 0` ↔ forward rolling (top tread moves +X, bottom −X).
    // With `r = pos − hub`, the spin velocity at the contact patch is
    //   `v_spin = [ω·r_y, 0, −ω·r_x]`,
    // which we add to `v_next` for the tangential motion below. The
    // sticky branch then reads "stuck = static + spin" so a spinning
    // hub on flat terrain produces a backward slip = forward friction
    // — same chain the CPU bristle integrator runs.
    // KNOWN HARDCODE (tire audit 2026-07-05): the built-in node-vs-
    // ground contact pins mu_static/mu_dynamic and the y = 0 plane —
    // terrain/material params flow only through the bristle feedback
    // stage, not this fallback.
    let mu_static: f32 = 1.0;
    let mu_dynamic: f32 = 0.85;
    let spin_omega = u.phys5.x;
    let r_x = p_next.x - hub.x;
    let r_y = p_next.y - hub.y;
    let spin_v_x = spin_omega * r_y;
    let spin_v_z = -spin_omega * r_x;
    if (p_next.y < 0.0) {
        let pen = -p_next.y;
        p_next.y = 0.0;
        let tx = (v_next.x + spin_v_x) * dt;
        let tz = (v_next.z + spin_v_z) * dt;
        let t_mag = sqrt(tx * tx + tz * tz);
        let cap_static = mu_static * pen;
        if (t_mag <= cap_static) {
            // Static — undo tangential motion.
            p_next.x = p.x;
            p_next.z = p.z;
            v_next.x = 0.0;
            v_next.z = 0.0;
        } else if (t_mag > 1.0e-12) {
            let keep = (t_mag - mu_dynamic * pen) / t_mag;
            p_next.x = p.x + tx * keep;
            p_next.z = p.z + tz * keep;
            v_next.x = (v_next.x + spin_v_x) * keep - spin_v_x;
            v_next.z = (v_next.z + spin_v_z) * keep - spin_v_z;
        }
        if (v_next.y < 0.0) { v_next.y = 0.0; }
    }

    pos_out[i] = vec4<f32>(p_next, 0.0);
    vel_out[i] = vec4<f32>(v_next, 0.0);
    force_out[i] = vec4<f32>(force, 0.0);
}
