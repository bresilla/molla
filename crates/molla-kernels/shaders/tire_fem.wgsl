// Volumetric tire tetra-FEM step.
//
// One invocation owns one node. To avoid float atomics/races, each node loops
// over all tetrahedra and accumulates only the force contributions for tets
// incident on that node. This is not the fastest layout for thousands of tires,
// but it is deterministic, simple to validate, and maps directly to the CPU
// `molla_vehicle::TetraTireFem` formulation.

struct Uniforms {
    // x=node_count, y=tet_count, z=bead_count, w=tread_count
    counts: vec4<u32>,
    // x=dt, y=young_modulus, z=poisson_ratio, w=pressure_pa
    phys0: vec4<f32>,
    // x=node_damping, y=bead_stiffness, z=floor_stiffness, w=floor_damping
    phys1: vec4<f32>,
    // x=pressure_area_per_tread_node, yzw=hub
    phys2: vec4<f32>,
    // x=gravity_y, yzw=reserved
    phys3: vec4<f32>,
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

fn inv3(m: mat3x3<f32>) -> mat3x3<f32> {
    let c0 = cross(m[1], m[2]);
    let c1 = cross(m[2], m[0]);
    let c2 = cross(m[0], m[1]);
    let d = max(abs(dot(m[0], c0)), 1.0e-8) * sign(dot(m[0], c0));
    return mat3x3<f32>(c0 / d, c1 / d, c2 / d);
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

    var force = vec3<f32>(0.0, m * u.phys3.x, 0.0) - u.phys1.x * v;

    // Tetra hyperelastic forces. Each node recomputes incident tet stress locally;
    // this intentionally avoids storage atomics.
    for (var t: u32 = 0u; t < u.counts.y; t = t + 1u) {
        let ib = t * 4u;
        let id0 = tet_indices[ib + 0u];
        let id1 = tet_indices[ib + 1u];
        let id2 = tet_indices[ib + 2u];
        let id3 = tet_indices[ib + 3u];

        var local: u32 = 4u;
        if (id0 == i) { local = 0u; }
        if (id1 == i) { local = 1u; }
        if (id2 == i) { local = 2u; }
        if (id3 == i) { local = 3u; }
        if (local == 4u) { continue; }

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
        let mu = tet_mu[t];
        let lambda = tet_lambda[t];
        let P = mu * (F - FinvT) + lambda * logJ * FinvT;

        let g_node = tet_grads[ib + local].xyz;
        let sg = P * g_node;
        force = force - tet_volume[t] * sg;
    }

    // Bead/rim coupling to the current hub pose.
    let hub = u.phys2.yzw;
    for (var b: u32 = 0u; b < u.counts.z; b = b + 1u) {
        if (bead_ids[b] == i) {
            let bead_target = hub + bead_offsets[b].xyz;
            force = force + u.phys1.y * (bead_target - p);
        }
    }

    // Pressure traction on outer tread nodes.
    for (var q: u32 = 0u; q < u.counts.w; q = q + 1u) {
        if (tread_ids[q] == i) {
            let radial = safe_normalize(vec3<f32>(p.x - hub.x, p.y - hub.y, 0.0), vec3<f32>(0.0, 1.0, 0.0));
            force = force + radial * (u.phys0.w * u.phys2.x);
        }
    }

    // Flat floor contact, matching the CPU viewer contact model.
    if (p.y < 0.0) {
        let pen = -p.y;
        let normal = max(0.0, u.phys1.z * pen - u.phys1.w * min(v.y, 0.0));
        force.y = force.y + normal;
        force.x = force.x - 320.0 * v.x;
        force.z = force.z - 320.0 * v.z;
    }

    var v_next = v + (force / m) * u.phys0.x;
    v_next = clamp(v_next, vec3<f32>(-6.0), vec3<f32>(6.0));
    var p_next = p + v_next * u.phys0.x;
    if (p_next.y < 0.0) {
        p_next.y = 0.0;
        if (v_next.y < 0.0) { v_next.y = 0.0; }
    }

    pos_out[i] = vec4<f32>(p_next, 0.0);
    vel_out[i] = vec4<f32>(v_next, 0.0);
    force_out[i] = vec4<f32>(force, 0.0);
}
