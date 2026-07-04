// C0 phase 2 — barrier-Newton contact on the GPU (f32 port of the
// molla-solvers `barrier_contact` f64 oracle; that oracle is the
// parity reference for every kernel here).
//
// Division of labor: the host drives the outer Newton loop and the
// Armijo backtracking with 4-byte scalar readbacks; everything heavy
// — gradient, Hessian-vector products, the PCG vector work and dot
// reductions, the conservative TOI, the trial-energy evaluation and
// the min-gap invariant scan — runs on device. All gathers are
// CSR-based and race-free (bit-deterministic, same discipline as
// cloth_diff.wgsl).
//
// Scalars buffer layout (all f32):
//   [0] dot result a   [1] dot result b   [2] cg alpha  [3] cg beta
//   [4] rr_old         [5] rr_init        [6] ccd alpha [7] energy
//   [8] min gap        [9] grad_inf

struct Uniforms {
    vertex_count: u32,
    spring_count: u32,
    obstacle_count: u32,
    _pad0: u32,
    h: f32,          // step size
    dhat: f32,
    kappa_scale: f32,
    line_alpha: f32, // trial α for x_try = x_iter + α·p
    gravity_x: f32,
    gravity_y: f32,
    gravity_z: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

// Topology / constants.
@group(1) @binding(0) var<storage, read> spring_a: array<u32>;
@group(1) @binding(1) var<storage, read> spring_b: array<u32>;
@group(1) @binding(2) var<storage, read> spring_rest: array<f32>;
@group(1) @binding(3) var<storage, read> spring_k: array<f32>;
@group(1) @binding(4) var<storage, read> inv_mass_unused: array<f32>; // reserved
@group(1) @binding(5) var<storage, read> mass: array<f32>;
@group(1) @binding(6) var<storage, read> csr_offsets: array<u32>;
@group(1) @binding(7) var<storage, read> csr_entries: array<u32>;
// Obstacles: two vec4 rows each —
//   row0 = (kind, p0, p1, p2)  kind 0 = half-space(n=p, offset=row1.x)
//                              kind 1 = sphere(center=p, radius=row1.x)
@group(1) @binding(8) var<storage, read> obstacles: array<vec4<f32>>;

// State (vec4-packed vec3s).
@group(2) @binding(0) var<storage, read_write> x_pos: array<vec4<f32>>;   // committed x
@group(2) @binding(1) var<storage, read_write> vel: array<vec4<f32>>;
@group(2) @binding(2) var<storage, read_write> x_tilde: array<vec4<f32>>;
@group(2) @binding(3) var<storage, read_write> x_iter: array<vec4<f32>>;  // Newton iterate
@group(2) @binding(4) var<storage, read_write> x_try: array<vec4<f32>>;   // line-search trial

// Newton / CG workspace.
@group(3) @binding(0) var<storage, read_write> grad: array<vec4<f32>>;
@group(3) @binding(1) var<storage, read_write> p_dir: array<vec4<f32>>;   // Newton direction
@group(3) @binding(2) var<storage, read_write> cg_r: array<vec4<f32>>;
@group(3) @binding(3) var<storage, read_write> cg_d: array<vec4<f32>>;
@group(3) @binding(4) var<storage, read_write> cg_hd: array<vec4<f32>>;
@group(3) @binding(5) var<storage, read_write> part: array<f32>;          // per-element partials
@group(3) @binding(6) var<storage, read_write> scalars: array<f32>;

const WG: u32 = 64u;

fn obstacle_distance(idx: u32, p: vec3<f32>) -> vec4<f32> {
    // Returns (grad_d.xyz, d).
    let r0 = obstacles[idx * 2u];
    let r1 = obstacles[idx * 2u + 1u];
    if (r0.x < 0.5) {
        let n = r0.yzw;
        return vec4<f32>(n, dot(n, p) - r1.x);
    }
    let d = p - r0.yzw;
    let len = max(length(d), 1.0e-9);
    return vec4<f32>(d / len, len - r1.x);
}

// b' and b'' of the cubic compact-support barrier.
fn barrier_d1(d: f32, kappa: f32) -> f32 {
    if (d >= u.dhat) { return 0.0; }
    let s = u.dhat - d;
    return -kappa * s * s / (u.dhat * u.dhat);
}
fn barrier_d2(d: f32, kappa: f32) -> f32 {
    if (d >= u.dhat) { return 0.0; }
    let s = u.dhat - d;
    return 2.0 * kappa * s / (u.dhat * u.dhat);
}
fn barrier_energy(d: f32, kappa: f32) -> f32 {
    if (d >= u.dhat) { return 0.0; }
    let s = u.dhat - d;
    return kappa * s * s * s / (3.0 * u.dhat * u.dhat);
}

@compute @workgroup_size(WG)
fn predict(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.vertex_count) { return; }
    let g = vec3<f32>(u.gravity_x, u.gravity_y, u.gravity_z);
    let xt = x_pos[i].xyz + vel[i].xyz * u.h + g * (u.h * u.h);
    x_tilde[i] = vec4<f32>(xt, 0.0);
    x_iter[i] = x_pos[i];
}

// ∇E at x_iter.
@compute @workgroup_size(WG)
fn gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.vertex_count) { return; }
    let xi = x_iter[i].xyz;
    var g = (xi - x_tilde[i].xyz) * mass[i];
    let h2 = u.h * u.h;
    for (var e = csr_offsets[i]; e < csr_offsets[i + 1u]; e = e + 1u) {
        let s = csr_entries[e];
        let a = spring_a[s];
        let b = spring_b[s];
        let other = select(a, b, i == a);
        let d = x_iter[other].xyz - xi;
        let len = max(length(d), 1.0e-9);
        // ∂E/∂x_i = −h²k(len − rest)·(d/len)  (pointing away from the
        // neighbour when stretched — mirror of the CPU sign layout).
        g = g - d / len * (h2 * spring_k[s] * (len - spring_rest[s]));
    }
    let kappa = u.kappa_scale * mass[i] / (u.h * u.h);
    for (var o = 0u; o < u.obstacle_count; o = o + 1u) {
        let gd = obstacle_distance(o, xi);
        g = g + gd.xyz * (h2 * barrier_d1(gd.w, kappa));
    }
    grad[i] = vec4<f32>(g, 0.0);
}

// y = (∇²E)·v with SPD projections; v lives in cg_d, y in cg_hd.
@compute @workgroup_size(WG)
fn hessian_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.vertex_count) { return; }
    let xi = x_iter[i].xyz;
    let vi = cg_d[i].xyz;
    var y = vi * mass[i];
    let h2 = u.h * u.h;
    for (var e = csr_offsets[i]; e < csr_offsets[i + 1u]; e = e + 1u) {
        let s = csr_entries[e];
        let a = spring_a[s];
        let b = spring_b[s];
        let other = select(a, b, i == a);
        let d = x_iter[other].xyz - xi;
        let len = max(length(d), 1.0e-9);
        let dir = d / len;
        let t = max(1.0 - spring_rest[s] / len, 0.0);
        let dv = vi - cg_d[other].xyz;
        let dv_par = dir * dot(dir, dv);
        y = y + ((dv - dv_par) * t + dv_par) * (h2 * spring_k[s]);
    }
    let kappa = u.kappa_scale * mass[i] / (u.h * u.h);
    for (var o = 0u; o < u.obstacle_count; o = o + 1u) {
        let gd = obstacle_distance(o, xi);
        let ddb = barrier_d2(gd.w, kappa);
        if (ddb > 0.0) {
            y = y + gd.xyz * (h2 * ddb * dot(gd.xyz, vi));
        }
    }
    cg_hd[i] = vec4<f32>(y, 0.0);
}

// ── CG plumbing (single-workgroup reductions; freeze discipline) ────

// cg_r = −grad ; cg_d = M⁻¹ cg_r ; p_dir = 0.
@compute @workgroup_size(WG)
fn cg_init(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.vertex_count) { return; }
    let r = -grad[i].xyz;
    cg_r[i] = vec4<f32>(r, 0.0);
    cg_d[i] = vec4<f32>(r / mass[i], 0.0);
    p_dir[i] = vec4<f32>(0.0);
}

var<workgroup> red: array<f32, 256>;

// scalars[4] = rz = Σ r·(M⁻¹r); scalars[5] = rr_init (‖r‖²).
@compute @workgroup_size(256)
fn cg_dot_init(@builtin(local_invocation_id) lid: vec3<u32>) {
    var rz = 0.0;
    var rr = 0.0;
    var i = lid.x;
    loop {
        if (i >= u.vertex_count) { break; }
        let r = cg_r[i].xyz;
        rz = rz + dot(r, r / mass[i]);
        rr = rr + dot(r, r);
        i = i + 256u;
    }
    red[lid.x] = rz;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) { red[lid.x] = red[lid.x] + red[lid.x + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid.x == 0u) { scalars[4] = red[0]; }
    workgroupBarrier();
    red[lid.x] = rr;
    workgroupBarrier();
    stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) { red[lid.x] = red[lid.x] + red[lid.x + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid.x == 0u) { scalars[5] = red[0]; }
}

// scalars[0] = d·Hd, then α = rz/dHd with the freeze guard.
@compute @workgroup_size(256)
fn cg_dot_dhd(@builtin(local_invocation_id) lid: vec3<u32>) {
    var acc = 0.0;
    var i = lid.x;
    loop {
        if (i >= u.vertex_count) { break; }
        acc = acc + dot(cg_d[i].xyz, cg_hd[i].xyz);
        i = i + 256u;
    }
    red[lid.x] = acc;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) { red[lid.x] = red[lid.x] + red[lid.x + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid.x == 0u) {
        let dhd = red[0];
        scalars[0] = dhd;
        // Freeze: f32 CG diverges if iterated past convergence
        // (measured twice in this codebase — APIC projection).
        let frozen = scalars[4] <= 1.0e-12 * max(scalars[5], 1.0e-30);
        if (!frozen && dhd > 1.0e-30) {
            scalars[2] = scalars[4] / dhd;
        } else {
            scalars[2] = 0.0;
        }
    }
}

// p += α d ; r −= α Hd.
@compute @workgroup_size(WG)
fn cg_update_pr(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.vertex_count) { return; }
    let alpha = scalars[2];
    p_dir[i] = vec4<f32>(p_dir[i].xyz + cg_d[i].xyz * alpha, 0.0);
    cg_r[i] = vec4<f32>(cg_r[i].xyz - cg_hd[i].xyz * alpha, 0.0);
}

// new rz → β = rz_new/rz_old (freeze-guarded); rz_old ← rz_new.
@compute @workgroup_size(256)
fn cg_dot_rz(@builtin(local_invocation_id) lid: vec3<u32>) {
    var acc = 0.0;
    var i = lid.x;
    loop {
        if (i >= u.vertex_count) { break; }
        let r = cg_r[i].xyz;
        acc = acc + dot(r, r / mass[i]);
        i = i + 256u;
    }
    red[lid.x] = acc;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) { red[lid.x] = red[lid.x] + red[lid.x + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid.x == 0u) {
        let rz_new = red[0];
        let frozen = rz_new <= 1.0e-12 * max(scalars[5], 1.0e-30);
        if (!frozen && scalars[4] > 1.0e-30) {
            scalars[3] = rz_new / scalars[4];
        } else {
            scalars[3] = 0.0;
        }
        scalars[4] = rz_new;
    }
}

// d = M⁻¹r + β d.
@compute @workgroup_size(WG)
fn cg_update_d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.vertex_count) { return; }
    cg_d[i] = vec4<f32>(cg_r[i].xyz / mass[i] + cg_d[i].xyz * scalars[3], 0.0);
}

// ── Line search machinery ───────────────────────────────────────────

// Conservative per-vertex TOI vs all obstacles along p_dir, min-reduced
// into scalars[6]. Exact for half-spaces; straight-line conservative
// for spheres; ×0.9 safety identical to the CPU oracle. THIS kernel is
// the additive-CCD TOI primitive later phases extend to VT/EE pairs.
@compute @workgroup_size(256)
fn ccd_alpha(@builtin(local_invocation_id) lid: vec3<u32>) {
    var a_min = 1.0;
    var i = lid.x;
    loop {
        if (i >= u.vertex_count) { break; }
        let xi = x_iter[i].xyz;
        let pi = p_dir[i].xyz;
        for (var o = 0u; o < u.obstacle_count; o = o + 1u) {
            let r0 = obstacles[o * 2u];
            let r1 = obstacles[o * 2u + 1u];
            if (r0.x < 0.5) {
                let n = r0.yzw;
                let gap = dot(n, xi) - r1.x;
                let approach = -dot(n, pi);
                if (approach > 0.0) {
                    a_min = min(a_min, 0.9 * gap / approach);
                }
            } else {
                let gap = length(xi - r0.yzw) - r1.x;
                let speed = length(pi);
                if (speed > 0.0) {
                    a_min = min(a_min, 0.9 * gap / speed);
                }
            }
        }
        i = i + 256u;
    }
    red[lid.x] = a_min;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) { red[lid.x] = min(red[lid.x], red[lid.x + stride]); }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid.x == 0u) {
        scalars[6] = clamp(red[0], 1.0e-9, 1.0);
    }
}

// scalars[1] = grad·p_dir (Armijo slope).
@compute @workgroup_size(256)
fn dot_grad_p(@builtin(local_invocation_id) lid: vec3<u32>) {
    var acc = 0.0;
    var i = lid.x;
    loop {
        if (i >= u.vertex_count) { break; }
        acc = acc + dot(grad[i].xyz, p_dir[i].xyz);
        i = i + 256u;
    }
    red[lid.x] = acc;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) { red[lid.x] = red[lid.x] + red[lid.x + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid.x == 0u) { scalars[1] = red[0]; }
}

// x_try = x_iter + line_alpha·p_dir.
@compute @workgroup_size(WG)
fn make_trial(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.vertex_count) { return; }
    x_try[i] = vec4<f32>(x_iter[i].xyz + p_dir[i].xyz * u.line_alpha, 0.0);
}

// Per-vertex energy partials of x_try (inertia + barriers) into part[i].
@compute @workgroup_size(WG)
fn energy_vertex(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.vertex_count) { return; }
    let xi = x_try[i].xyz;
    let dx = xi - x_tilde[i].xyz;
    var e = 0.5 * mass[i] * dot(dx, dx);
    let kappa = u.kappa_scale * mass[i] / (u.h * u.h);
    for (var o = 0u; o < u.obstacle_count; o = o + 1u) {
        let gd = obstacle_distance(o, xi);
        e = e + u.h * u.h * barrier_energy(gd.w, kappa);
    }
    part[i] = e;
}

// Per-spring energy partials into part[vertex_count + s].
@compute @workgroup_size(WG)
fn energy_spring(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if (s >= u.spring_count) { return; }
    let d = x_try[spring_b[s]].xyz - x_try[spring_a[s]].xyz;
    let c = length(d) - spring_rest[s];
    part[u.vertex_count + s] = u.h * u.h * 0.5 * spring_k[s] * c * c;
}

// Sum partials into scalars[7]; also min-gap of x_try into scalars[8]
// and ‖grad‖_∞ into scalars[9] (used by the host for convergence and
// the invariant WITHOUT extra passes).
@compute @workgroup_size(256)
fn reduce_energy_gap(@builtin(local_invocation_id) lid: vec3<u32>) {
    let total = u.vertex_count + u.spring_count;
    var e = 0.0;
    var i = lid.x;
    loop {
        if (i >= total) { break; }
        e = e + part[i];
        i = i + 256u;
    }
    red[lid.x] = e;
    workgroupBarrier();
    var stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) { red[lid.x] = red[lid.x] + red[lid.x + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid.x == 0u) { scalars[7] = red[0]; }
    workgroupBarrier();
    // min gap over x_try
    var g_min = 3.4e38;
    i = lid.x;
    loop {
        if (i >= u.vertex_count) { break; }
        for (var o = 0u; o < u.obstacle_count; o = o + 1u) {
            g_min = min(g_min, obstacle_distance(o, x_try[i].xyz).w);
        }
        i = i + 256u;
    }
    red[lid.x] = g_min;
    workgroupBarrier();
    stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) { red[lid.x] = min(red[lid.x], red[lid.x + stride]); }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid.x == 0u) { scalars[8] = red[0]; }
    workgroupBarrier();
    var g_inf = 0.0;
    i = lid.x;
    loop {
        if (i >= u.vertex_count) { break; }
        let g = abs(grad[i].xyz);
        g_inf = max(g_inf, max(g.x, max(g.y, g.z)));
        i = i + 256u;
    }
    red[lid.x] = g_inf;
    workgroupBarrier();
    stride = 128u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) { red[lid.x] = max(red[lid.x], red[lid.x + stride]); }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid.x == 0u) { scalars[9] = red[0]; }
}

// Accept the trial: x_iter ← x_try.
@compute @workgroup_size(WG)
fn accept_trial(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.vertex_count) { return; }
    x_iter[i] = x_try[i];
}

// Commit the step: v ← (x_iter − x)/h ; x ← x_iter.
@compute @workgroup_size(WG)
fn commit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.vertex_count) { return; }
    vel[i] = vec4<f32>((x_iter[i].xyz - x_pos[i].xyz) / u.h, 0.0);
    x_pos[i] = x_iter[i];
}
