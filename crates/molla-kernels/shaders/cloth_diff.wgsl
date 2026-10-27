// Differentiable mass-spring cloth (INSPECTION.md F15 / DiffsimCloth
// GPU adjoint) — forward + manual reverse-mode kernels.
//
// Forward, per step (semi-implicit Euler, gather form):
//   f_i   = Σ_{s ∋ i} sign_is · k_s (L_s − rest_s) d̂_s
//   v_out = v_in + (f/m + g)·dt      (pinned particles: v = 0)
//   p_out = p_in + v_out·dt
//
// Reverse, per step (given ∂J/∂p_out, ∂J/∂v_out):
//   â_v   = adj_v_out + dt·adj_p_out
//   adj_p_in[i] = adj_p_out[i]
//               + dt Σ_{s=(a,b) ∋ i} (∂f_a/∂p_i)ᵀ â_v[a]/m_a
//                                  + (∂f_b/∂p_i)ᵀ â_v[b]/m_b
//   adj_v_in = â_v
//   adj_k[s] += dt (L−rest) d̂ · (â_v[b]/m_b − â_v[a]/m_a)
// with the standard spring Jacobian
//   J_s = k[(1 − rest/L)(I − d̂d̂ᵀ) + d̂d̂ᵀ],
//   ∂f_a/∂p_a = −J, ∂f_a/∂p_b = +J (f_b = −f_a).
//
// Everything is GATHER-form (per-particle loop over incident springs
// via CSR; per-spring kernel for adj_k) — no atomics, so the forward
// is bit-deterministic and finite differences on the GPU forward are
// a valid oracle for the analytic reverse pass.

struct Uniforms {
    particle_count: u32,
    spring_count: u32,
    step: u32,          // current step index (trajectory addressing)
    tet_count: u32,
    dt: f32,
    gravity_x: f32,
    gravity_y: f32,
    gravity_z: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

// Static topology / parameters.
@group(1) @binding(0) var<storage, read> spring_a: array<u32>;
@group(1) @binding(1) var<storage, read> spring_b: array<u32>;
@group(1) @binding(2) var<storage, read> spring_rest: array<f32>;
@group(1) @binding(3) var<storage, read_write> spring_k: array<f32>;
@group(1) @binding(4) var<storage, read> inv_mass: array<f32>; // 0 = pinned
@group(1) @binding(5) var<storage, read> csr_offsets: array<u32>;
@group(1) @binding(6) var<storage, read> csr_entries: array<u32>; // spring id
// Softbody volume term (per-tet energy ½·k_v·(V − V₀)²):
@group(1) @binding(7) var<storage, read> tet_ids: array<vec4<u32>>;
@group(1) @binding(8) var<storage, read> tet_v0: array<f32>;
@group(1) @binding(9) var<storage, read_write> tet_kv: array<f32>;
@group(1) @binding(10) var<storage, read> tet_csr_offsets: array<u32>;
@group(1) @binding(11) var<storage, read> tet_csr_entries: array<u32>; // tet id

// Trajectory: positions/velocities for steps 0..=T, laid out as
// step-major slabs of `particle_count` vec4s.
@group(2) @binding(0) var<storage, read_write> traj_p: array<vec4<f32>>;
@group(2) @binding(1) var<storage, read_write> traj_v: array<vec4<f32>>;

// Adjoint state (per particle) + per-spring stiffness gradient.
@group(3) @binding(0) var<storage, read_write> adj_p: array<vec4<f32>>;
@group(3) @binding(1) var<storage, read_write> adj_v: array<vec4<f32>>;
@group(3) @binding(2) var<storage, read_write> adj_k: array<f32>;
// â_v staging: written by `backward_hat`, read (race-free) by the
// gather passes — in-place adjoint updates would otherwise race on
// neighbours' values mid-dispatch.
@group(3) @binding(3) var<storage, read_write> adj_vh: array<vec4<f32>>;
@group(3) @binding(4) var<storage, read_write> adj_kv: array<f32>;

fn spring_force_on(i: u32, s: u32, p_slab: u32) -> vec3<f32> {
    let a = spring_a[s];
    let b = spring_b[s];
    let pa = traj_p[p_slab + a].xyz;
    let pb = traj_p[p_slab + b].xyz;
    let d = pb - pa;
    let len = length(d);
    if (len < 1.0e-9) {
        return vec3<f32>(0.0);
    }
    let dir = d / len;
    let f_a = spring_k[s] * (len - spring_rest[s]) * dir;
    if (i == a) {
        return f_a;
    }
    return -f_a;
}

// Signed tet volume and the four volume gradients
// (∂V/∂p_j; g_a = −(g_b+g_c+g_d); Hessian blocks are hat matrices of
// edges — see the reverse pass).
fn tet_volume(pa: vec3<f32>, pb: vec3<f32>, pc: vec3<f32>, pd: vec3<f32>) -> f32 {
    return dot(cross(pb - pa, pc - pa), pd - pa) / 6.0;
}

// Volume force on the LOCAL vertex `slot` (0..3) of tet t at slab.
fn tet_volume_force(slot: u32, t: u32, p_slab: u32) -> vec3<f32> {
    let ids = tet_ids[t];
    let pa = traj_p[p_slab + ids.x].xyz;
    let pb = traj_p[p_slab + ids.y].xyz;
    let pc = traj_p[p_slab + ids.z].xyz;
    let pd = traj_p[p_slab + ids.w].xyz;
    let v = tet_volume(pa, pb, pc, pd);
    let s = -tet_kv[t] * (v - tet_v0[t]);
    let g_b = cross(pc - pa, pd - pa) / 6.0;
    let g_c = cross(pd - pa, pb - pa) / 6.0;
    let g_d = cross(pb - pa, pc - pa) / 6.0;
    if (slot == 0u) {
        return s * (-(g_b + g_c + g_d));
    }
    if (slot == 1u) {
        return s * g_b;
    }
    if (slot == 2u) {
        return s * g_c;
    }
    return s * g_d;
}

fn tet_local_slot(t: u32, i: u32) -> u32 {
    let ids = tet_ids[t];
    if (ids.x == i) { return 0u; }
    if (ids.y == i) { return 1u; }
    if (ids.z == i) { return 2u; }
    return 3u;
}

@compute @workgroup_size(64)
fn forward_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.particle_count) { return; }
    let slab_in = u.step * u.particle_count;
    let slab_out = (u.step + 1u) * u.particle_count;
    let w = inv_mass[i];
    if (w == 0.0) {
        traj_v[slab_out + i] = vec4<f32>(0.0);
        traj_p[slab_out + i] = traj_p[slab_in + i];
        return;
    }
    var f = vec3<f32>(0.0);
    for (var e = csr_offsets[i]; e < csr_offsets[i + 1u]; e = e + 1u) {
        f = f + spring_force_on(i, csr_entries[e], slab_in);
    }
    if (u.tet_count > 0u) {
        for (var e = tet_csr_offsets[i]; e < tet_csr_offsets[i + 1u]; e = e + 1u) {
            let t = tet_csr_entries[e];
            f = f + tet_volume_force(tet_local_slot(t, i), t, slab_in);
        }
    }
    let g = vec3<f32>(u.gravity_x, u.gravity_y, u.gravity_z);
    let v_new = traj_v[slab_in + i].xyz + (f * w + g) * u.dt;
    let p_new = traj_p[slab_in + i].xyz + v_new * u.dt;
    traj_v[slab_out + i] = vec4<f32>(v_new, 0.0);
    traj_p[slab_out + i] = vec4<f32>(p_new, 0.0);
}

// J_s · x for the spring Jacobian evaluated from slab positions.
fn spring_jac_mul(s: u32, p_slab: u32, x: vec3<f32>) -> vec3<f32> {
    let a = spring_a[s];
    let b = spring_b[s];
    let d = traj_p[p_slab + b].xyz - traj_p[p_slab + a].xyz;
    let len = length(d);
    if (len < 1.0e-9) {
        return vec3<f32>(0.0);
    }
    let dir = d / len;
    let k = spring_k[s];
    let t = 1.0 - spring_rest[s] / len;
    let x_par = dir * dot(dir, x);
    return k * (t * (x - x_par) + x_par);
}

// Pass 1 of the reverse step: â_v = adj_v + dt·adj_p per particle,
// staged into `adj_vh` so the gather passes read stable values.
@compute @workgroup_size(64)
fn backward_hat(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.particle_count) { return; }
    if (inv_mass[i] == 0.0) {
        adj_vh[i] = vec4<f32>(0.0);
        return;
    }
    adj_vh[i] = vec4<f32>(adj_v[i].xyz + u.dt * adj_p[i].xyz, 0.0);
}

// Pass 3 of the reverse step: consumes adj_p (∂J/∂p_out) + the staged
// â_v and REPLACES adj_p/adj_v with (∂J/∂p_in, ∂J/∂v_in). `u.step` is
// the input slab index (the same the forward step read from).
@compute @workgroup_size(64)
fn backward_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.particle_count) { return; }
    let slab_in = u.step * u.particle_count;
    let w_i = inv_mass[i];
    if (w_i == 0.0) {
        // Pinned: p_out = p_in (adjoint passes through), v frozen.
        adj_v[i] = vec4<f32>(0.0);
        return;
    }
    let a_v_hat_i = adj_vh[i].xyz;
    var a_p_in = adj_p[i].xyz;
    // Gather (∂f/∂p_i)ᵀ contributions. J is symmetric; sign table:
    //   i == a: ∂f_a/∂p_a = −J ; ∂f_b/∂p_a = +J
    //   i == b: ∂f_a/∂p_b = +J ; ∂f_b/∂p_b = −J
    // Both cases fold to x = w_other·â_v[other] − w_i·â_v[i].
    for (var e = csr_offsets[i]; e < csr_offsets[i + 1u]; e = e + 1u) {
        let s = csr_entries[e];
        let a = spring_a[s];
        let b = spring_b[s];
        let other = select(a, b, i == a);
        let w_other = inv_mass[other];
        var x = -a_v_hat_i * w_i;
        if (w_other > 0.0) {
            x = x + adj_vh[other].xyz * w_other;
        }
        a_p_in = a_p_in + u.dt * spring_jac_mul(s, slab_in, x);
    }
    // Volume-term adjoint. For tet t with x_j = w_j·â_v[j]:
    //   (∂f_j/∂p_i)ᵀ x_j summed over j
    //   = −k_v [ g_i·(Σ_j g_j·x_j) + (V−V₀)·(Σ_j ∂g_i/∂p_j x_j) ]
    // using Hessian symmetry ∂g_j/∂p_i = (∂g_i/∂p_j)ᵀ and the
    // closed-form blocks ∂g_b/∂p_c = −hat(pd−pa)/6 etc. (volume is
    // trilinear, so each block is an edge cross-product matrix).
    if (u.tet_count > 0u) {
        for (var e = tet_csr_offsets[i]; e < tet_csr_offsets[i + 1u]; e = e + 1u) {
            let t = tet_csr_entries[e];
            let ids = tet_ids[t];
            let pa = traj_p[slab_in + ids.x].xyz;
            let pb = traj_p[slab_in + ids.y].xyz;
            let pc = traj_p[slab_in + ids.z].xyz;
            let pd = traj_p[slab_in + ids.w].xyz;
            let v = tet_volume(pa, pb, pc, pd);
            let g_b = cross(pc - pa, pd - pa) / 6.0;
            let g_c = cross(pd - pa, pb - pa) / 6.0;
            let g_d = cross(pb - pa, pc - pa) / 6.0;
            let g_a = -(g_b + g_c + g_d);
            let x_a = adj_vh[ids.x].xyz * inv_mass[ids.x];
            let x_b = adj_vh[ids.y].xyz * inv_mass[ids.y];
            let x_c = adj_vh[ids.z].xyz * inv_mass[ids.z];
            let x_d = adj_vh[ids.w].xyz * inv_mass[ids.w];
            let s_t = dot(g_a, x_a) + dot(g_b, x_b) + dot(g_c, x_c) + dot(g_d, x_d);
            // y_i = Σ_j (∂g_i/∂p_j) x_j — apply the Hessian row for
            // the LOCAL slot of i. Blocks (excluding a-row, derived
            // by translation invariance):
            //   ∂g_b/∂p_c = −hat(pd−pa)/6 ; ∂g_b/∂p_d = hat(pc−pa)/6
            //   ∂g_c/∂p_d = −hat(pb−pa)/6 ; ∂g_c/∂p_b = hat(pd−pa)/6
            //   ∂g_d/∂p_b = −hat(pc−pa)/6 ; ∂g_d/∂p_c = hat(pb−pa)/6
            //   diagonal blocks are zero; a-row/col from −sums.
            let ea = pd - pa;
            let eb = pc - pa;
            let ec = pb - pa;
            let slot = tet_local_slot(t, i);
            var y = vec3<f32>(0.0);
            if (slot == 1u) {
                // g_b row: cols c, d, a.
                y = -cross(ea, x_c) / 6.0 + cross(eb, x_d) / 6.0
                    + (cross(ea, x_a) - cross(eb, x_a)) / 6.0;
            } else if (slot == 2u) {
                // g_c row: cols d, b, a.
                y = -cross(ec, x_d) / 6.0 + cross(ea, x_b) / 6.0
                    + (cross(ec, x_a) - cross(ea, x_a)) / 6.0;
            } else if (slot == 3u) {
                // g_d row: cols b, c, a.
                y = -cross(eb, x_b) / 6.0 + cross(ec, x_c) / 6.0
                    + (cross(eb, x_a) - cross(ec, x_a)) / 6.0;
            } else {
                // a row = −(b+c+d rows applied to the same x).
                let yb = -cross(ea, x_c) / 6.0 + cross(eb, x_d) / 6.0
                    + (cross(ea, x_a) - cross(eb, x_a)) / 6.0;
                let yc2 = -cross(ec, x_d) / 6.0 + cross(ea, x_b) / 6.0
                    + (cross(ec, x_a) - cross(ea, x_a)) / 6.0;
                let yd2 = -cross(eb, x_b) / 6.0 + cross(ec, x_c) / 6.0
                    + (cross(eb, x_a) - cross(ec, x_a)) / 6.0;
                y = -(yb + yc2 + yd2);
            }
            let kv = tet_kv[t];
            a_p_in = a_p_in
                - u.dt * kv
                    * (s_t * tet_row_gradient(slot, g_a, g_b, g_c, g_d)
                        + (v - tet_v0[t]) * y);
        }
    }
    adj_p[i] = vec4<f32>(a_p_in, 0.0);
    adj_v[i] = vec4<f32>(a_v_hat_i, 0.0);
}

fn tet_row_gradient(slot: u32, g_a: vec3<f32>, g_b: vec3<f32>, g_c: vec3<f32>, g_d: vec3<f32>) -> vec3<f32> {
    if (slot == 0u) { return g_a; }
    if (slot == 1u) { return g_b; }
    if (slot == 2u) { return g_c; }
    return g_d;
}

// Per-tet volume-stiffness gradient (pass 2b — after backward_hat,
// before backward_step): ∂J/∂k_v[t] += −dt (V−V₀) Σ_j g_j·(w_j â_j).
@compute @workgroup_size(64)
fn backward_volume_stiffness(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= u.tet_count) { return; }
    let slab_in = u.step * u.particle_count;
    let ids = tet_ids[t];
    let pa = traj_p[slab_in + ids.x].xyz;
    let pb = traj_p[slab_in + ids.y].xyz;
    let pc = traj_p[slab_in + ids.z].xyz;
    let pd = traj_p[slab_in + ids.w].xyz;
    let v = tet_volume(pa, pb, pc, pd);
    let g_b = cross(pc - pa, pd - pa) / 6.0;
    let g_c = cross(pd - pa, pb - pa) / 6.0;
    let g_d = cross(pb - pa, pc - pa) / 6.0;
    let g_a = -(g_b + g_c + g_d);
    let s_t = dot(g_a, adj_vh[ids.x].xyz * inv_mass[ids.x])
        + dot(g_b, adj_vh[ids.y].xyz * inv_mass[ids.y])
        + dot(g_c, adj_vh[ids.z].xyz * inv_mass[ids.z])
        + dot(g_d, adj_vh[ids.w].xyz * inv_mass[ids.w]);
    adj_kv[t] = adj_kv[t] - u.dt * (v - tet_v0[t]) * s_t;
}

// Pass 2 of the reverse step: per-spring stiffness gradient. Reads
// the staged â_v (dispatch after `backward_hat`, before
// `backward_step` overwrites adj_p/adj_v).
@compute @workgroup_size(64)
fn backward_stiffness(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if (s >= u.spring_count) { return; }
    let slab_in = u.step * u.particle_count;
    let a = spring_a[s];
    let b = spring_b[s];
    let d = traj_p[slab_in + b].xyz - traj_p[slab_in + a].xyz;
    let len = length(d);
    if (len < 1.0e-9) { return; }
    let dir = d / len;
    let dfdk_a = (len - spring_rest[s]) * dir; // ∂f_a/∂k (f_b = −f_a)
    var acc = 0.0;
    if (inv_mass[a] > 0.0) {
        acc = acc + dot(dfdk_a, adj_vh[a].xyz) * inv_mass[a];
    }
    if (inv_mass[b] > 0.0) {
        acc = acc - dot(dfdk_a, adj_vh[b].xyz) * inv_mass[b];
    }
    adj_k[s] = adj_k[s] + u.dt * acc;
}
