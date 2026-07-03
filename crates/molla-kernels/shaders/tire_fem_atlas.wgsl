// Fused tire-FEM atlas shader.
//
// One invocation owns one node in a packed multi-tire atlas. The shader keeps
// the same physical node update as `tire_fem.wgsl` but loads per-tire scalar
// state from storage rows instead of a single uniform block. All topology
// indices are atlas-global; `node_tire_indices`, `param_rows`, and
// `offset_rows` route each node to its tire-local counts/ranges.

struct AtlasParamRow {
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
    phys3: vec4<f32>,
    // x=xpbd_bead_compliance, y=semi_implicit_stiffness_scale,
    // z=sidewall_ply_stretch_limit, w=sidewall_ply_compliance
    phys4: vec4<f32>,
};

struct AtlasOffsetRow {
    // x=node_offset, y=node_count, z=node_workgroup_offset, w=node_workgroup_count
    node: vec4<u32>,
    // x=tet_offset, y=tet_count, zw=reserved
    tet: vec4<u32>,
    // x=tread_offset, y=tread_count, z=bead_offset, w=bead_count
    contact: vec4<u32>,
    // x=edge_offset, y=edge_count, z=dihedral_offset, w=dihedral_count
    xpbd: vec4<u32>,
};

struct AtlasHubParamRow {
    // x=spin inertia (kg·m², > 0 enables the resident spin DOF — F2a/b),
    // y=hub_floor_active, z=hub_floor_y, w=dt
    cfg: vec4<f32>,
    // x=bead_stiffness, y=bead_damping, z=hub_mass, w=hub_damping
    mech: vec4<f32>,
    // x=downforce, y=gravity_y, z=drive torque (N·m), w=brake torque (N·m)
    load: vec4<f32>,
    // xyz=external hub force (N), w=lateral damping coefficient
    extf: vec4<f32>,
};

struct BrushWheelOutput {
    forces: vec4<f32>,
    diag1: vec4<f32>,
    diag2: vec4<f32>,
    diag3: vec4<f32>,
    diag4: vec4<f32>,
    diag5: vec4<f32>,
    world_force: vec4<f32>,
};

struct BrushHubUniforms {
    // x=tire_count, yzw=reserved
    counts: vec4<u32>,
    // x=force_scale, yzw=reserved
    params: vec4<f32>,
};

struct ContactReductionOutputRows {
    diagnostics0: vec4<f32>,
    diagnostics1: vec4<f32>,
    world_force_sinkage: vec4<f32>,
    world_moment_water: vec4<f32>,
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

@group(0) @binding(15) var<storage, read>      tet_fiber_dir: array<vec4<f32>>;
@group(0) @binding(16) var<storage, read>      tet_fiber_k1:  array<f32>;
@group(0) @binding(17) var<storage, read>      tet_fiber_k2:  array<f32>;
@group(0) @binding(18) var<storage, read>      node_tet_offsets: array<u32>;
@group(0) @binding(19) var<storage, read>      node_tet_entries: array<u32>;
@group(0) @binding(20) var<storage, read_write> hub_state: array<vec4<f32>>;
@group(0) @binding(21) var<storage, read>      tet_fiber_dir_b: array<vec4<f32>>;
@group(0) @binding(22) var<storage, read>      tet_fiber_k1_b:  array<f32>;
@group(0) @binding(23) var<storage, read>      tet_fiber_k2_b:  array<f32>;
@group(0) @binding(24) var<storage, read>      xpbd_edge_indices: array<vec4<u32>>;
@group(0) @binding(25) var<storage, read>      xpbd_edge_params:  array<vec4<f32>>;
@group(0) @binding(26) var<storage, read>      node_edge_offsets: array<u32>;
@group(0) @binding(27) var<storage, read>      node_edge_entries: array<u32>;
@group(0) @binding(28) var<storage, read>      tet_fiber_dir_c: array<vec4<f32>>;
@group(0) @binding(29) var<storage, read>      tet_fiber_k1_c:  array<f32>;
@group(0) @binding(30) var<storage, read>      tet_fiber_k2_c:  array<f32>;
@group(0) @binding(31) var<storage, read>      tet_fiber_dir_d: array<vec4<f32>>;
@group(0) @binding(32) var<storage, read>      tet_fiber_k1_d:  array<f32>;
@group(0) @binding(33) var<storage, read>      tet_fiber_k2_d:  array<f32>;
@group(0) @binding(34) var<storage, read>      xpbd_dihedral_indices: array<vec4<u32>>;
@group(0) @binding(35) var<storage, read>      xpbd_dihedral_params:  array<vec4<f32>>;
@group(0) @binding(36) var<storage, read>      node_dihedral_offsets: array<u32>;
@group(0) @binding(37) var<storage, read>      node_dihedral_entries: array<u32>;

@group(0) @binding(38) var<storage, read>      node_tire_indices:  array<u32>;
@group(0) @binding(39) var<storage, read>      tet_tire_indices:   array<u32>;
@group(0) @binding(40) var<storage, read>      bead_tire_indices:  array<u32>;
@group(0) @binding(41) var<storage, read>      tread_tire_indices: array<u32>;
@group(0) @binding(42) var<storage, read>      param_rows:  array<AtlasParamRow>;
@group(0) @binding(43) var<storage, read>      offset_rows: array<AtlasOffsetRow>;
@group(0) @binding(44) var<storage, read>      hub_param_rows: array<AtlasHubParamRow>;
@group(0) @binding(45) var<storage, read>      brush_hub_outputs: array<BrushWheelOutput>;
@group(0) @binding(46) var<storage, read_write> brush_hub_param_rows: array<AtlasHubParamRow>;
@group(0) @binding(47) var<storage, read>      contact_hub_outputs: array<ContactReductionOutputRows>;
@group(0) @binding(48) var<storage, read>      feedback_hub_forces: array<vec4<f32>>;

@group(1) @binding(0) var<uniform> brush_hub_u: BrushHubUniforms;

const ATLAS_HUB_WG: u32 = 128u;
var<workgroup> atlas_hub_partial_fx: array<f32, ATLAS_HUB_WG>;
var<workgroup> atlas_hub_partial_fy: array<f32, ATLAS_HUB_WG>;
var<workgroup> atlas_hub_partial_fz: array<f32, ATLAS_HUB_WG>;

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

fn inv3(m: mat3x3<f32>) -> mat3x3<f32> {
    let c0 = cross(m[1], m[2]);
    let c1 = cross(m[2], m[0]);
    let c2 = cross(m[0], m[1]);
    let d_raw = dot(m[0], c0);
    let d = max(abs(d_raw), 1.0e-8) * sign(d_raw);
    return mat3x3<f32>(c0 / d, c1 / d, c2 / d);
}

fn add_fiber_stress(P_in: mat3x3<f32>, F: mat3x3<f32>, a0: vec3<f32>, k1: f32, k2: f32) -> mat3x3<f32> {
    var P = P_in;
    if (k1 <= 0.0) {
        return P;
    }
    let Fa0 = F * a0;
    let i4 = dot(Fa0, Fa0);
    if (i4 <= 1.0) {
        return P;
    }
    let stretch = i4 - 1.0;
    var exp_factor: f32 = 1.0;
    if (k2 > 0.0) {
        exp_factor = exp(min(k2 * stretch * stretch, 30.0));
    }
    let coeff = 2.0 * k1 * stretch * exp_factor;
    P[0][0] = P[0][0] + coeff * Fa0.x * a0.x;
    P[1][0] = P[1][0] + coeff * Fa0.x * a0.y;
    P[2][0] = P[2][0] + coeff * Fa0.x * a0.z;
    P[0][1] = P[0][1] + coeff * Fa0.y * a0.x;
    P[1][1] = P[1][1] + coeff * Fa0.y * a0.y;
    P[2][1] = P[2][1] + coeff * Fa0.y * a0.z;
    P[0][2] = P[0][2] + coeff * Fa0.z * a0.x;
    P[1][2] = P[1][2] + coeff * Fa0.z * a0.y;
    P[2][2] = P[2][2] + coeff * Fa0.z * a0.z;
    return P;
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

fn quad_dihedral_angle(x0: vec3<f32>, x1: vec3<f32>, x2: vec3<f32>, x3: vec3<f32>) -> f32 {
    let n0 = safe_normalize(cross(x1 - x0, x2 - x0), vec3<f32>(1.0, 0.0, 0.0));
    let n1 = safe_normalize(cross(x2 - x0, x3 - x0), vec3<f32>(1.0, 0.0, 0.0));
    return acos(clamp(dot(n0, n1), -1.0, 1.0));
}

fn quad_dihedral_grad(local: u32, axis: u32, x0: vec3<f32>, x1: vec3<f32>, x2: vec3<f32>, x3: vec3<f32>) -> f32 {
    let eps = 1.0e-5;
    var p0 = x0;
    var p1 = x1;
    var p2 = x2;
    var p3 = x3;
    if (local == 0u) {
        if (axis == 0u) { p0.x = p0.x + eps; }
        if (axis == 1u) { p0.y = p0.y + eps; }
        if (axis == 2u) { p0.z = p0.z + eps; }
    }
    if (local == 1u) {
        if (axis == 0u) { p1.x = p1.x + eps; }
        if (axis == 1u) { p1.y = p1.y + eps; }
        if (axis == 2u) { p1.z = p1.z + eps; }
    }
    if (local == 2u) {
        if (axis == 0u) { p2.x = p2.x + eps; }
        if (axis == 1u) { p2.y = p2.y + eps; }
        if (axis == 2u) { p2.z = p2.z + eps; }
    }
    if (local == 3u) {
        if (axis == 0u) { p3.x = p3.x + eps; }
        if (axis == 1u) { p3.y = p3.y + eps; }
        if (axis == 2u) { p3.z = p3.z + eps; }
    }
    let plus = quad_dihedral_angle(p0, p1, p2, p3);
    p0 = x0;
    p1 = x1;
    p2 = x2;
    p3 = x3;
    if (local == 0u) {
        if (axis == 0u) { p0.x = p0.x - eps; }
        if (axis == 1u) { p0.y = p0.y - eps; }
        if (axis == 2u) { p0.z = p0.z - eps; }
    }
    if (local == 1u) {
        if (axis == 0u) { p1.x = p1.x - eps; }
        if (axis == 1u) { p1.y = p1.y - eps; }
        if (axis == 2u) { p1.z = p1.z - eps; }
    }
    if (local == 2u) {
        if (axis == 0u) { p2.x = p2.x - eps; }
        if (axis == 1u) { p2.y = p2.y - eps; }
        if (axis == 2u) { p2.z = p2.z - eps; }
    }
    if (local == 3u) {
        if (axis == 0u) { p3.x = p3.x - eps; }
        if (axis == 1u) { p3.y = p3.y - eps; }
        if (axis == 2u) { p3.z = p3.z - eps; }
    }
    let minus = quad_dihedral_angle(p0, p1, p2, p3);
    return clamp((plus - minus) / (2.0 * eps), -1.0e4, 1.0e4);
}

fn quad_dihedral_grad_vec(local: u32, x0: vec3<f32>, x1: vec3<f32>, x2: vec3<f32>, x3: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        quad_dihedral_grad(local, 0u, x0, x1, x2, x3),
        quad_dihedral_grad(local, 1u, x0, x1, x2, x3),
        quad_dihedral_grad(local, 2u, x0, x1, x2, x3),
    );
}

@compute @workgroup_size(64)
fn step_nodes_atlas_contract(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&pos_in)) {
        return;
    }

    let tire = node_tire_indices[i];
    let params = param_rows[tire];
    let offsets = offset_rows[tire];
    let local_i = i - offsets.node.x;
    if (local_i >= params.counts.x) {
        return;
    }

    let p = pos_in[i].xyz;
    let v = vel_in[i].xyz;
    let m = mass_in[i];
    if (m <= 0.0) {
        pos_out[i] = vec4<f32>(p, 0.0);
        vel_out[i] = vec4<f32>(0.0);
        force_out[i] = vec4<f32>(0.0);
        return;
    }

    var force = vec3<f32>(0.0, m * params.phys3.x, 0.0);
    var node_stiffness_diag: f32 = 0.0;

    let csr_lo = node_tet_offsets[i];
    let csr_hi = node_tet_offsets[i + 1u];
    let tet_begin = offsets.tet.x;
    let tet_end = offsets.tet.x + params.counts.y;
    for (var e: u32 = csr_lo; e < csr_hi; e = e + 1u) {
        let entry = node_tet_entries[e];
        let t = entry >> 2u;
        let local = entry & 3u;
        if (t < tet_begin || t >= tet_end) {
            continue;
        }
        let ib = t * 4u;

        var f0 = vec3<f32>(0.0);
        var f1 = vec3<f32>(0.0);
        var f2 = vec3<f32>(0.0);
        for (var a: u32 = 0u; a < 4u; a = a + 1u) {
            let id = tet_indices[ib + a];
            let x = pos_in[id].xyz;
            let g = tet_grads[ib + a].xyz;
            f0 = f0 + x.x * g;
            f1 = f1 + x.y * g;
            f2 = f2 + x.z * g;
        }

        let F = transpose(mat3x3<f32>(f0, f1, f2));
        let FinvT = transpose(inv3(F));
        let J = max(det3(F), 0.20);
        let logJ = log(J);
        let mu = tet_mu[t] + params.phys3.z;
        let lambda = tet_lambda[t];
        var P = mu * (F - FinvT) + lambda * logJ * FinvT;

        P = add_fiber_stress(P, F, tet_fiber_dir[t].xyz, tet_fiber_k1[t], tet_fiber_k2[t]);
        P = add_fiber_stress(P, F, tet_fiber_dir_b[t].xyz, tet_fiber_k1_b[t], tet_fiber_k2_b[t]);
        P = add_fiber_stress(P, F, tet_fiber_dir_c[t].xyz, tet_fiber_k1_c[t], tet_fiber_k2_c[t]);
        P = add_fiber_stress(P, F, tet_fiber_dir_d[t].xyz, tet_fiber_k1_d[t], tet_fiber_k2_d[t]);

        let g_node = tet_grads[ib + local].xyz;
        let sg = P * g_node;
        force = force - tet_volume[t] * sg;

        let grad_norm2 = dot(g_node, g_node);
        let fibre_tangent = abs(tet_fiber_k1[t]) * (1.0 + abs(tet_fiber_k2[t]))
            + abs(tet_fiber_k1_b[t]) * (1.0 + abs(tet_fiber_k2_b[t]))
            + abs(tet_fiber_k1_c[t]) * (1.0 + abs(tet_fiber_k2_c[t]))
            + abs(tet_fiber_k1_d[t]) * (1.0 + abs(tet_fiber_k2_d[t]));
        node_stiffness_diag = node_stiffness_diag
            + tet_volume[t] * (abs(mu) + abs(lambda) + fibre_tangent) * grad_norm2;
    }

    let hub = hub_state[tire * 2u].xyz;
    let cavity_per_bead = params.phys3.y;
    let bead_begin = offsets.contact.z;
    let bead_end = offsets.contact.z + params.counts.z;
    for (var b: u32 = bead_begin; b < bead_end; b = b + 1u) {
        if (bead_ids[b] == i) {
            let bead_target = hub + bead_offsets[b].xyz;
            force = force + params.phys1.y * (bead_target - p);
            node_stiffness_diag = node_stiffness_diag + max(params.phys1.y, 0.0);
            if (cavity_per_bead > 0.0) {
                let radial = safe_normalize(
                    vec3<f32>(p.x - hub.x, p.y - hub.y, 0.0),
                    vec3<f32>(1.0, 0.0, 0.0),
                );
                force = force + radial * cavity_per_bead;
            }
        }
    }

    let tread_begin = offsets.contact.x;
    let tread_end = offsets.contact.x + params.counts.w;
    for (var q: u32 = tread_begin; q < tread_end; q = q + 1u) {
        if (tread_ids[q] == i) {
            let radial = safe_normalize(vec3<f32>(p.x - hub.x, p.y - hub.y, 0.0), vec3<f32>(0.0, 1.0, 0.0));
            force = force + radial * (params.phys0.w * params.phys2.x);
        }
    }

    let dt = params.phys0.x;
    let damping = params.phys1.x;
    let semi_implicit_stiffness = max(params.phys4.y, 0.0) * max(node_stiffness_diag, 0.0);
    let denom = m + damping * dt + semi_implicit_stiffness * dt * dt;
    var v_next = (m * v + force * dt) / denom;
    v_next = clamp(v_next, vec3<f32>(-8.0), vec3<f32>(8.0));
    var p_next = p + v_next * dt;

    let xpbd_volume_compliance = params.phys3.w;
    if (xpbd_volume_compliance <= 1.0e20) {
        let alpha = max(xpbd_volume_compliance, 0.0) / max(dt * dt, 1.0e-12);
        var dp_sum = vec3<f32>(0.0);
        var dp_count = 0.0;
        for (var e2: u32 = csr_lo; e2 < csr_hi; e2 = e2 + 1u) {
            let entry = node_tet_entries[e2];
            let t = entry >> 2u;
            let local = entry & 3u;
            if (t < tet_begin || t >= tet_end) {
                continue;
            }
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

    let xpbd_bead_compliance = params.phys4.x;
    if (xpbd_bead_compliance <= 1.0e20) {
        let alpha_b = max(xpbd_bead_compliance, 0.0) / max(dt * dt, 1.0e-12);
        for (var b2: u32 = bead_begin; b2 < bead_end; b2 = b2 + 1u) {
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

    if (params.counts2.x > 0u) {
        let edge_lo = node_edge_offsets[i];
        let edge_hi = node_edge_offsets[i + 1u];
        let edge_begin = offsets.xpbd.x;
        let edge_end = offsets.xpbd.x + params.counts2.x;
        var dp_edge_sum = vec3<f32>(0.0);
        var dp_edge_count = 0.0;
        for (var ee: u32 = edge_lo; ee < edge_hi; ee = ee + 1u) {
            let packed = node_edge_entries[ee];
            let eidx = packed >> 1u;
            if (eidx < edge_begin || eidx >= edge_end) {
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

    if (params.counts2.y > 0u) {
        let diag_lo = node_dihedral_offsets[i];
        let diag_hi = node_dihedral_offsets[i + 1u];
        let diag_begin = offsets.xpbd.z;
        let diag_end = offsets.xpbd.z + params.counts2.y;
        var dp_diag_sum = vec3<f32>(0.0);
        var dp_diag_count = 0.0;
        for (var dd: u32 = diag_lo; dd < diag_hi; dd = dd + 1u) {
            let packed = node_dihedral_entries[dd];
            let didx = packed >> 2u;
            if (didx < diag_begin || didx >= diag_end) {
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
            let g0 = quad_dihedral_grad_vec(0u, x0, x1, x2, x3);
            let g1 = quad_dihedral_grad_vec(1u, x0, x1, x2, x3);
            let g2 = quad_dihedral_grad_vec(2u, x0, x1, x2, x3);
            let g3 = quad_dihedral_grad_vec(3u, x0, x1, x2, x3);
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

    // Stage (f4) — atlas version of the tensile sidewall ply-strain XPBD
    // guardrail. Uses atlas-global node/tet/fibre rows and the same local
    // gather correction as the single-tire shader.
    let ply_limit = max(params.phys4.z, 1.01);
    let ply_compliance = params.phys4.w;
    if (ply_compliance <= 1.0e20 && ply_limit > 1.0) {
        let alpha_p = max(ply_compliance, 0.0) / max(dt * dt, 1.0e-12);
        var dp_ply_sum = vec3<f32>(0.0);
        var dp_ply_count = 0.0;
        for (var pe: u32 = csr_lo; pe < csr_hi; pe = pe + 1u) {
            let entry = node_tet_entries[pe];
            let t = entry >> 2u;
            let local = entry & 3u;
            if (t < tet_begin || t >= tet_end) {
                continue;
            }
            let ib = t * 4u;
            if (tet_fiber_k1[t] <= 0.0 || tet_fiber_k1_b[t] <= 0.0 || tet_fiber_k1_c[t] <= 0.0 || tet_fiber_k1_d[t] <= 0.0) {
                continue;
            }
            let id0 = tet_indices[ib + 0u];
            let id1 = tet_indices[ib + 1u];
            let id2 = tet_indices[ib + 2u];
            let id3 = tet_indices[ib + 3u];
            let centroid = (rest_pos[id0].xyz + rest_pos[id1].xyz + rest_pos[id2].xyz + rest_pos[id3].xyz) * 0.25;
            let radial = safe_normalize(vec3<f32>(centroid.x - hub.x, centroid.y - hub.y, 0.0), vec3<f32>(1.0, 0.0, 0.0));
            let axial = vec3<f32>(0.0, 0.0, 1.0);
            if (abs(dot(tet_fiber_dir_c[t].xyz, radial)) < 0.85 || abs(dot(tet_fiber_dir_d[t].xyz, axial)) < 0.85) {
                continue;
            }
            var pf0 = vec3<f32>(0.0);
            var pf1 = vec3<f32>(0.0);
            var pf2 = vec3<f32>(0.0);
            for (var a: u32 = 0u; a < 4u; a = a + 1u) {
                let id = tet_indices[ib + a];
                let x = pos_in[id].xyz;
                let g = tet_grads[ib + a].xyz;
                pf0 = pf0 + x.x * g;
                pf1 = pf1 + x.y * g;
                pf2 = pf2 + x.z * g;
            }
            let Fp = transpose(mat3x3<f32>(pf0, pf1, pf2));
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
                let g0s = dot(tet_grads[ib + 0u].xyz, dir);
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

    let mu_static: f32 = 1.0;
    let mu_dynamic: f32 = 0.85;
    if (p_next.y < 0.0) {
        let pen = -p_next.y;
        p_next.y = 0.0;
        let tx = v_next.x * dt;
        let tz = v_next.z * dt;
        let t_mag = sqrt(tx * tx + tz * tz);
        let cap_static = mu_static * pen;
        if (t_mag <= cap_static) {
            p_next.x = p.x;
            p_next.z = p.z;
            v_next.x = 0.0;
            v_next.z = 0.0;
        } else if (t_mag > 1.0e-12) {
            let keep = (t_mag - mu_dynamic * pen) / t_mag;
            p_next.x = p.x + tx * keep;
            p_next.z = p.z + tz * keep;
            v_next.x = v_next.x * keep;
            v_next.z = v_next.z * keep;
        }
        if (v_next.y < 0.0) { v_next.y = 0.0; }
    }

    pos_out[i] = vec4<f32>(p_next, 0.0);
    vel_out[i] = vec4<f32>(v_next, 0.0);
    force_out[i] = vec4<f32>(force, 0.0);
}

@compute @workgroup_size(128)
fn hub_reduce_atlas(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tire = wid.x;
    if (tire >= arrayLength(&offset_rows)) {
        return;
    }
    let i = lid.x;
    let offsets = offset_rows[tire];
    let hp = hub_param_rows[tire];
    let bead_begin = offsets.contact.z;
    let bead_count = offsets.contact.w;
    let hub_row = tire * 2u;
    let hub = hub_state[hub_row];          // (hub_x, hub_y, hub_z, hub_v_y)
    let hub_v_xz = hub_state[hub_row + 1u]; // (hub_v_x, hub_v_z, _, _)
    let bead_k = hp.mech.x;
    let bead_c = hp.mech.y;

    var local_x: f32 = 0.0;
    var local_y: f32 = 0.0;
    var local_z: f32 = 0.0;
    var local_bead = i;
    loop {
        if (local_bead >= bead_count) {
            break;
        }
        let b = bead_begin + local_bead;
        let id = bead_ids[b];
        let offset = bead_offsets[b].xyz;
        let pos = pos_in[id].xyz;
        let vel = vel_in[id].xyz;
        let bead_target = vec3<f32>(hub.x, hub.y, hub.z) + offset;
        let delta = bead_target - pos;
        let rel_v = vec3<f32>(hub_v_xz.x, hub.w, hub_v_xz.y) - vel;
        let force_on_bead = bead_k * delta + bead_c * rel_v;
        local_x = local_x + force_on_bead.x;
        local_y = local_y + force_on_bead.y;
        local_z = local_z + force_on_bead.z;
        local_bead = local_bead + ATLAS_HUB_WG;
    }
    atlas_hub_partial_fx[i] = local_x;
    atlas_hub_partial_fy[i] = local_y;
    atlas_hub_partial_fz[i] = local_z;
    workgroupBarrier();

    var stride: u32 = ATLAS_HUB_WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (i < stride) {
            atlas_hub_partial_fx[i] = atlas_hub_partial_fx[i] + atlas_hub_partial_fx[i + stride];
            atlas_hub_partial_fy[i] = atlas_hub_partial_fy[i] + atlas_hub_partial_fy[i + stride];
            atlas_hub_partial_fz[i] = atlas_hub_partial_fz[i] + atlas_hub_partial_fz[i + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (i == 0u) {
        let downforce = hp.load.x;
        let gravity_y = hp.load.y;
        let hub_mass = max(hp.mech.z, 1.0e-6);
        let hub_damp = max(hp.mech.w, 0.0);
        let lat_damp = max(hp.extf.w, 0.0);
        let dt = max(hp.cfg.w, 1.0e-9);

        let reaction_x = -atlas_hub_partial_fx[0];
        let reaction_y = -atlas_hub_partial_fy[0];
        let reaction_z = -atlas_hub_partial_fz[0];

        let force_x = reaction_x + hp.extf.x;
        let force_y = reaction_y - downforce + gravity_y * hub_mass + hp.extf.y;
        let force_z = reaction_z + hp.extf.z;

        let denom_y = hub_mass + hub_damp * dt;
        var v_new_y = (hub_mass * hub.w + force_y * dt) / denom_y;
        v_new_y = clamp(v_new_y, -2.5, 2.5);

        let denom_xz = hub_mass + lat_damp * dt;
        var v_new_x = (hub_mass * hub_v_xz.x + force_x * dt) / denom_xz;
        var v_new_z = (hub_mass * hub_v_xz.y + force_z * dt) / denom_xz;
        v_new_x = clamp(v_new_x, -25.0, 25.0);
        v_new_z = clamp(v_new_z, -25.0, 25.0);

        var x_new = hub.x + v_new_x * dt;
        var y_new = hub.y + v_new_y * dt;
        var z_new = hub.z + v_new_z * dt;
        if (hp.cfg.y > 0.5) {
            let hub_floor = hp.cfg.z;
            if (y_new < hub_floor) {
                y_new = hub_floor;
                if (v_new_y < 0.0) { v_new_y = 0.0; }
            }
        }
        hub_state[hub_row] = vec4<f32>(x_new, y_new, z_new, v_new_y);

        // Resident spin DOF (F2a/b) — same integration as
        // tire_fem_hub.wgsl; passthrough when inactive.
        var omega = hub_v_xz.z;
        var angle = hub_v_xz.w;
        if (hp.cfg.x > 0.0) {
            let inertia = max(hp.cfg.x, 1.0e-6);
            let brake_soft = omega / (abs(omega) + 0.5);
            let alpha = (hp.load.z - hp.load.w * brake_soft) / inertia;
            omega = omega + alpha * dt;
            if (!(omega == omega)) { omega = 0.0; }
            angle = angle + omega * dt;
            if (!(angle == angle)) { angle = 0.0; }
        }
        hub_state[hub_row + 1u] = vec4<f32>(v_new_x, v_new_z, omega, angle);
    }
}

@compute @workgroup_size(64)
fn hub_params_from_brush_outputs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tire = gid.x;
    let n = min(brush_hub_u.counts.x, arrayLength(&brush_hub_param_rows));
    if (tire >= n) {
        return;
    }
    var row = brush_hub_param_rows[tire];
    let brush = brush_hub_outputs[tire];
    let scale = clamp(brush_hub_u.params.x, -10.0, 10.0);
    row.extf = vec4<f32>(
        row.extf.x + brush.world_force.x * scale,
        row.extf.y + brush.world_force.y * scale,
        row.extf.z + brush.world_force.z * scale,
        row.extf.w,
    );
    brush_hub_param_rows[tire] = row;
}

@compute @workgroup_size(64)
fn hub_params_from_contact_reduction_outputs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tire = gid.x;
    let n = min(brush_hub_u.counts.x, arrayLength(&brush_hub_param_rows));
    if (tire >= n) {
        return;
    }
    var row = brush_hub_param_rows[tire];
    let force = contact_hub_outputs[tire].world_force_sinkage.xyz;
    let scale = clamp(brush_hub_u.params.x, -10.0, 10.0);
    row.extf = vec4<f32>(
        row.extf.x + force.x * scale,
        row.extf.y + force.y * scale,
        row.extf.z + force.z * scale,
        row.extf.w,
    );
    brush_hub_param_rows[tire] = row;
}

@compute @workgroup_size(64)
fn hub_params_from_feedback_force_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tire = gid.x;
    let n = min(brush_hub_u.counts.x, arrayLength(&brush_hub_param_rows));
    if (tire >= n || tire >= arrayLength(&offset_rows)) {
        return;
    }
    let offsets = offset_rows[tire];
    let start = offsets.node.x;
    let count = offsets.node.y;
    let end = min(start + count, arrayLength(&feedback_hub_forces));
    var force = vec3<f32>(0.0, 0.0, 0.0);
    var i = start;
    loop {
        if (i >= end) {
            break;
        }
        force = force + feedback_hub_forces[i].xyz;
        i = i + 1u;
    }
    var row = brush_hub_param_rows[tire];
    let scale = clamp(brush_hub_u.params.x, -10.0, 10.0);
    row.extf = vec4<f32>(
        row.extf.x + force.x * scale,
        row.extf.y + force.y * scale,
        row.extf.z + force.z * scale,
        row.extf.w,
    );
    brush_hub_param_rows[tire] = row;
}
