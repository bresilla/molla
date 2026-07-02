// GPU source-guard shader for the matrix-free Newton tire-FEM debug timestep.
//
// This is the WGSL counterpart to
// `molla_vehicle::real_tire::matrix_free_newton_correction` (CPU path) and to
// `TireFemMatrixFreeNewtonGpuContract` (host-side dispatch shape).
//
// Each entry point owns one piece of the matrix-free Newton step. The host
// orchestrates the GMRES + line search loop on top of these dispatches:
//
//   1. `residual_eval`        — per-node residual for the current candidate
//                                position vector.
//   2. `jv_probe`             — finite-difference J·v product, given a base
//                                position and a perturbation direction.
//   3. `diagonal_extract`     — Jacobi diagonal sample of `J` along the
//                                canonical DoF basis (one workgroup per DoF).
//   4. `apply_correction`     — `pos_out = pos_in + alpha * dx` for a chosen
//                                line-search alpha.
//
// Today the contract status is `WgslSourceGuardOnly`: the kernels-side source
// guard verifies the entry-point set, but no host yet drives these dispatches
// in the persistent batch. Live dispatch will flip the status to `Live` —
// see `crates/molla-kernels/src/tire_fem.rs::tire_fem_matrix_free_newton_gpu_contract`.

struct NewtonUniforms {
    // x = node_count, y = tet_count, z = dof_count, w = reserved
    counts: vec4<u32>,
    // x = dt, y = node_damping, z = jv_epsilon, w = line_search_alpha
    phys0: vec4<f32>,
    // x = direction_dof_index (used by diagonal_extract and jv_probe),
    // y = krylov_iteration_budget, z = reserved, w = reserved
    indices: vec4<u32>,
};

@group(0) @binding(0) var<uniform> u: NewtonUniforms;

@group(0) @binding(1) var<storage, read>       rest_pos:     array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       x0_pos:       array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>       v0_vel:       array<vec4<f32>>;
@group(0) @binding(4) var<storage, read>       mass_in:      array<f32>;
@group(0) @binding(5) var<storage, read>       external_in:  array<vec4<f32>>;
@group(0) @binding(6) var<storage, read>       tet_indices:  array<u32>;
@group(0) @binding(7) var<storage, read>       tet_grads:    array<vec4<f32>>;
@group(0) @binding(8) var<storage, read>       tet_volume:   array<f32>;
@group(0) @binding(9) var<storage, read>       tet_mu:       array<f32>;
@group(0) @binding(10) var<storage, read>      tet_lambda:   array<f32>;

@group(0) @binding(11) var<storage, read>      candidate_pos: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read>      direction_in:  array<vec4<f32>>;

@group(0) @binding(13) var<storage, read_write> residual_out: array<vec4<f32>>;
@group(0) @binding(14) var<storage, read_write> jv_out:       array<vec4<f32>>;
@group(0) @binding(15) var<storage, read_write> diagonal_out: array<vec4<f32>>;
@group(0) @binding(16) var<storage, read_write> pos_out:      array<vec4<f32>>;

const WORKGROUP_SIZE: u32 = 64u;

// ----------------------------------------------------------------------------
// Internal force evaluation
//
// Mirrors the compressible Neo-Hookean tetra force in `tire_fem.wgsl`. One
// invocation collects the contribution from every tet that touches the
// current node by scanning the tet index buffer. The atlas/CSR path is
// available in the production stepper; here the matrix-free Newton kernel
// keeps the simpler O(node·tet) scan because each Jv probe and residual
// evaluation needs only the per-node sum, not adjacency-aware throughput.
// ----------------------------------------------------------------------------

fn tetra_force_on_node(node_id: u32, positions: ptr<function, array<vec3<f32>, 1024>>) -> vec3<f32> {
    var f = vec3<f32>(0.0, 0.0, 0.0);
    let tet_count = u.counts.y;
    for (var t: u32 = 0u; t < tet_count; t = t + 1u) {
        let base = t * 4u;
        let i0 = tet_indices[base + 0u];
        let i1 = tet_indices[base + 1u];
        let i2 = tet_indices[base + 2u];
        let i3 = tet_indices[base + 3u];
        if (i0 != node_id && i1 != node_id && i2 != node_id && i3 != node_id) {
            continue;
        }
        let p0 = (*positions)[i0];
        let p1 = (*positions)[i1];
        let p2 = (*positions)[i2];
        let p3 = (*positions)[i3];
        let g0 = tet_grads[base + 0u].xyz;
        let g1 = tet_grads[base + 1u].xyz;
        let g2 = tet_grads[base + 2u].xyz;
        let g3 = tet_grads[base + 3u].xyz;
        let F = mat3x3<f32>(
            p0 * g0.x + p1 * g1.x + p2 * g2.x + p3 * g3.x,
            p0 * g0.y + p1 * g1.y + p2 * g2.y + p3 * g3.y,
            p0 * g0.z + p1 * g1.z + p2 * g2.z + p3 * g3.z,
        );
        let det_f = determinant(F);
        let safe_j = max(det_f, 1.0e-4);
        let mu = tet_mu[t];
        let lambda = tet_lambda[t];
        let log_j = log(safe_j);
        let inv_j = 1.0 / safe_j;
        let inv_f = inv_j * transpose(adjugate3(F));
        let p_stress = mu * F + (lambda * log_j - mu) * inv_f;
        let volume = tet_volume[t];
        var contrib = vec3<f32>(0.0, 0.0, 0.0);
        if (i0 == node_id) { contrib = -volume * (p_stress * g0); }
        else if (i1 == node_id) { contrib = -volume * (p_stress * g1); }
        else if (i2 == node_id) { contrib = -volume * (p_stress * g2); }
        else if (i3 == node_id) { contrib = -volume * (p_stress * g3); }
        f = f + contrib;
    }
    return f;
}

fn adjugate3(m: mat3x3<f32>) -> mat3x3<f32> {
    let a = m[0]; let b = m[1]; let c = m[2];
    let r0 = vec3<f32>(b.y * c.z - b.z * c.y, b.z * c.x - b.x * c.z, b.x * c.y - b.y * c.x);
    let r1 = vec3<f32>(c.y * a.z - c.z * a.y, c.z * a.x - c.x * a.z, c.x * a.y - c.y * a.x);
    let r2 = vec3<f32>(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
    return mat3x3<f32>(r0, r1, r2);
}

fn residual_at_node(node_id: u32, positions: ptr<function, array<vec3<f32>, 1024>>) -> vec3<f32> {
    let dt = u.phys0.x;
    let damping = u.phys0.y;
    let mass = max(mass_in[node_id], 1.0e-9);
    let x0 = x0_pos[node_id].xyz;
    let v0 = v0_vel[node_id].xyz;
    let x = (*positions)[node_id];
    let velocity = (x - x0) / max(dt, 1.0e-9);
    let internal = tetra_force_on_node(node_id, positions);
    let external_force = external_in[node_id].xyz;
    return mass * (velocity - v0) - dt * (internal + external_force - damping * velocity);
}

// ----------------------------------------------------------------------------
// Entry: residual_eval
//
// One invocation per node. Writes `residual_out[node]` = R(node) given the
// current `candidate_pos`. This is the per-node-vec3 form of
// `patch_newton_residual_with_masses` on the CPU.
// ----------------------------------------------------------------------------

@compute @workgroup_size(WORKGROUP_SIZE)
fn residual_eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.counts.x) {
        return;
    }
    var positions: array<vec3<f32>, 1024>;
    let node_count = min(u.counts.x, 1024u);
    for (var n: u32 = 0u; n < node_count; n = n + 1u) {
        positions[n] = candidate_pos[n].xyz;
    }
    let r = residual_at_node(i, &positions);
    residual_out[i] = vec4<f32>(r, 0.0);
}

// ----------------------------------------------------------------------------
// Entry: jv_probe
//
// Matrix-free J·v = (R(x + eps·v) - R(x)) / eps. One invocation per node.
// The host primes `candidate_pos = x`, `direction_in = v`, and reads back
// `jv_out` after dispatch.
// ----------------------------------------------------------------------------

@compute @workgroup_size(WORKGROUP_SIZE)
fn jv_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.counts.x) {
        return;
    }
    let eps = u.phys0.z;
    var base_positions: array<vec3<f32>, 1024>;
    var perturbed_positions: array<vec3<f32>, 1024>;
    let node_count = min(u.counts.x, 1024u);
    for (var n: u32 = 0u; n < node_count; n = n + 1u) {
        let p = candidate_pos[n].xyz;
        base_positions[n] = p;
        perturbed_positions[n] = p + eps * direction_in[n].xyz;
    }
    let r_base = residual_at_node(i, &base_positions);
    let r_pert = residual_at_node(i, &perturbed_positions);
    jv_out[i] = vec4<f32>((r_pert - r_base) / max(eps, 1.0e-20), 0.0);
}

// ----------------------------------------------------------------------------
// Entry: diagonal_extract
//
// Probes the Jacobi diagonal of J by setting one DoF direction at a time.
// The host dispatches `dof_count = 3 * node_count` workgroups, each writing
// `diagonal_out[node]` packed as the per-axis diagonal of J for that node.
// This is matrix-free: each invocation reuses `residual_at_node` against a
// per-axis perturbation rather than assembling a dense Jacobian.
// ----------------------------------------------------------------------------

@compute @workgroup_size(WORKGROUP_SIZE)
fn diagonal_extract(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.counts.x) {
        return;
    }
    let eps = u.phys0.z;
    var positions: array<vec3<f32>, 1024>;
    let node_count = min(u.counts.x, 1024u);
    for (var n: u32 = 0u; n < node_count; n = n + 1u) {
        positions[n] = candidate_pos[n].xyz;
    }
    let r_base = residual_at_node(i, &positions);
    var diag = vec3<f32>(0.0, 0.0, 0.0);
    for (var axis: u32 = 0u; axis < 3u; axis = axis + 1u) {
        var saved = positions[i];
        var perturbed = saved;
        if (axis == 0u) { perturbed.x = perturbed.x + eps; }
        else if (axis == 1u) { perturbed.y = perturbed.y + eps; }
        else { perturbed.z = perturbed.z + eps; }
        positions[i] = perturbed;
        let r_pert = residual_at_node(i, &positions);
        positions[i] = saved;
        let column = (r_pert - r_base) / max(eps, 1.0e-20);
        if (axis == 0u) { diag.x = column.x; }
        else if (axis == 1u) { diag.y = column.y; }
        else { diag.z = column.z; }
    }
    diagonal_out[i] = vec4<f32>(diag, 0.0);
}

// ----------------------------------------------------------------------------
// Entry: apply_correction
//
// `pos_out = candidate_pos + alpha * direction_in`. Used by the host line
// search after GMRES, then again as the final accepted update before
// projection/contact passes run on the host.
// ----------------------------------------------------------------------------

@compute @workgroup_size(WORKGROUP_SIZE)
fn apply_correction(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.counts.x) {
        return;
    }
    let alpha = u.phys0.w;
    let p = candidate_pos[i].xyz;
    let dir = direction_in[i].xyz;
    pos_out[i] = vec4<f32>(p + alpha * dir, 0.0);
}
