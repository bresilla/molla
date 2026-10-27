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
    tri_count: u32,
    h: f32,          // step size
    dhat: f32,
    kappa_scale: f32,
    line_alpha: f32, // trial α for x_try = x_iter + α·p
    gravity_x: f32,
    gravity_y: f32,
    gravity_z: f32,
    mu_k: f32,       // kinetic Coulomb μ (0 = friction off)
    edge_count: u32,
    pair_capacity: u32,
    tet_count: u32,
    kappa_elastic: u32, // 1 = elasticity-inclusive κ
    mu_s_scale: f32, // μ_s = scale·μ_k (≥ 1)
    epsv: f32,       // stick-well width as velocity (e = ε_v·h)
    strain_limit: f32, // hard strain bound (0 = off)
    strain_scale: f32, // κ_s multiplier on k·rest²
    // Spatial-hash pair pruning (phase 5c). cell_size ≥ max primitive
    // extent + 3·d̂ so a primitive spans ≤ 2 cells per axis and a
    // single-ring probe is COMPLETE; hash collisions only ADD
    // candidates (Teschner hash), never lose them.
    grid_cell: f32,
    grid_slots: u32,   // power of two; 0 = brute force
    grid_cap: u32,     // bucket capacity
    // 1 = pair discovery from EXTERNAL candidate lists (LBVH broad
    // phase, step-granularity) instead of the per-iterate hash grid.
    pair_source: u32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

// Topology / constants.
@group(1) @binding(0) var<storage, read> spring_a: array<u32>;
@group(1) @binding(1) var<storage, read> spring_b: array<u32>;
@group(1) @binding(2) var<storage, read> spring_rest: array<f32>;
@group(1) @binding(3) var<storage, read> spring_k: array<f32>;
// Per-vertex Σ incident spring k — the local elastic eigen-stiffness
// bound for the elasticity-inclusive κ (TOG 43.6 scheme).
@group(1) @binding(4) var<storage, read> vertex_ksum: array<f32>;
@group(1) @binding(5) var<storage, read> mass: array<f32>;
@group(1) @binding(6) var<storage, read> csr_offsets: array<u32>;
@group(1) @binding(7) var<storage, read> csr_entries: array<u32>;
// Obstacles: two vec4 rows each —
//   row0 = (kind, p0, p1, p2)  kind 0 = half-space(n=p, offset=row1.x)
//                              kind 1 = sphere(center=p, radius=row1.x)
@group(1) @binding(8) var<storage, read> obstacles: array<vec4<f32>>;
// Self-contact mesh (phase 3b): triangles as vec4<u32> rows (a,b,c,0),
// unique edges as vec4<u32> rows (a,b,0,0), and the pinned mask.
@group(1) @binding(9) var<storage, read> tris: array<vec4<u32>>;
@group(1) @binding(10) var<storage, read> edges: array<vec4<u32>>;
@group(1) @binding(11) var<storage, read> pinned: array<u32>;
// Volumetric tets (phase 5a): ids as vec4<u32>, per-tet (kv, v0).
@group(1) @binding(12) var<storage, read> tet_ids: array<vec4<u32>>;
@group(1) @binding(13) var<storage, read> tet_kv_v0: array<vec2<f32>>;

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
// Self-contact pairs, rebuilt on device every Newton iterate:
// meta = (kind 0=VT/1=EE, first index, second index, 0),
// wts  = VT (w0,w1,w2,0) / EE (s,t,0,0) — clamped closest-point
// weights FROZEN at discovery (exact-a.e. gradients, same contract
// as the CPU oracle).
@group(3) @binding(7) var<storage, read_write> pair_meta: array<vec4<u32>>;
@group(3) @binding(8) var<storage, read_write> pair_wts: array<vec4<f32>>;
// [0] = pair count (atomic), [1] = overflow flag.
@group(3) @binding(9) var<storage, read_write> pair_count: array<atomic<u32>>;
// Lagged friction rows (phase 5a), one per (vertex, obstacle):
// (anchor.xyz, λ). λ = 0 marks an inactive row. Rebuilt at step start
// from the committed positions — the IPC semi-implicit lag.
@group(3) @binding(10) var<storage, read_write> fric_lag: array<vec4<f32>>;
// Hash-grid buckets: [0..slots) tri counts, items at slot*cap. The
// edge grid lives at offset `grid_slots` / `grid_slots*cap`.
@group(3) @binding(11) var<storage, read_write> grid_count: array<atomic<u32>>;
@group(3) @binding(12) var<storage, read_write> grid_items: array<u32>;
// External broad-phase candidates (pair_source = 1): counts in
// cand_meta[0]/[1], mixed-prim VT pairs (vertex id, V + tri id) and
// edge pairs. Filled by buffer copies from an LbvhBroadPhase.
@group(3) @binding(13) var<storage, read> cand_meta: array<u32>;
@group(3) @binding(14) var<storage, read> cand_vt_a: array<u32>;
@group(3) @binding(15) var<storage, read> cand_vt_b: array<u32>;
@group(3) @binding(16) var<storage, read> cand_ee_a: array<u32>;
@group(3) @binding(17) var<storage, read> cand_ee_b: array<u32>;
// Lagged PAIR friction (cloth-on-cloth), frozen at step start:
// meta = (kind, a, b, 0); wts = frozen closest-point weights;
// nl = (contact normal, λ); anchors = 4 rows per pair (stencil vertex
// positions at lag time). Count in pair_count[2], overflow [3].
@group(3) @binding(18) var<storage, read_write> lag_meta: array<vec4<u32>>;
@group(3) @binding(19) var<storage, read_write> lag_wts: array<vec4<f32>>;
@group(3) @binding(20) var<storage, read_write> lag_nl: array<vec4<f32>>;
@group(3) @binding(21) var<storage, read_write> lag_anchors: array<vec4<f32>>;

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

// Ericson §5.1.5 — closest point on tri (a,b,c) to p, barycentric.
fn tri_closest(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { return vec3<f32>(1.0, 0.0, 0.0); }
    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { return vec3<f32>(0.0, 1.0, 0.0); }
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / (d1 - d3);
        return vec3<f32>(1.0 - v, v, 0.0);
    }
    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { return vec3<f32>(0.0, 0.0, 1.0); }
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / (d2 - d6);
        return vec3<f32>(1.0 - w, 0.0, w);
    }
    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return vec3<f32>(0.0, 1.0 - w, w);
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    return vec3<f32>(1.0 - v - w, v, w);
}

// Ericson §5.1.9 — closest params (s, t) between two segments.
fn seg_closest(p1: vec3<f32>, q1: vec3<f32>, p2: vec3<f32>, q2: vec3<f32>) -> vec2<f32> {
    let d1 = q1 - p1;
    let d2 = q2 - p2;
    let r = p1 - p2;
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);
    let tiny = 1.0e-12;
    if (a <= tiny && e <= tiny) { return vec2<f32>(0.0, 0.0); }
    if (a <= tiny) { return vec2<f32>(0.0, clamp(f / e, 0.0, 1.0)); }
    let c = dot(d1, r);
    if (e <= tiny) { return vec2<f32>(clamp(-c / a, 0.0, 1.0), 0.0); }
    let b = dot(d1, d2);
    let denom = a * e - b * b;
    var s = 0.0;
    if (denom > tiny) { s = clamp((b * f - c * e) / denom, 0.0, 1.0); }
    var t = (b * s + f) / e;
    if (t < 0.0) {
        t = 0.0;
        s = clamp(-c / a, 0.0, 1.0);
    } else if (t > 1.0) {
        t = 1.0;
        s = clamp((b - c) / a, 0.0, 1.0);
    }
    return vec2<f32>(s, t);
}

// Pair difference vector (first side − second side) and the 4-vertex
// stencil at positions from `use_try` (x_try) or x_iter.
fn pair_diff(p: u32, use_try: bool) -> vec3<f32> {
    let m = pair_meta[p];
    let w = pair_wts[p];
    if (m.x == 0u) {
        let t = tris[m.z];
        var cp: vec3<f32>;
        var pv: vec3<f32>;
        if (use_try) {
            cp = x_try[t.x].xyz * w.x + x_try[t.y].xyz * w.y + x_try[t.z].xyz * w.z;
            pv = x_try[m.y].xyz;
        } else {
            cp = x_iter[t.x].xyz * w.x + x_iter[t.y].xyz * w.y + x_iter[t.z].xyz * w.z;
            pv = x_iter[m.y].xyz;
        }
        return pv - cp;
    }
    let e1 = edges[m.y];
    let e2 = edges[m.z];
    var pa: vec3<f32>;
    var pb: vec3<f32>;
    if (use_try) {
        pa = x_try[e1.x].xyz * (1.0 - w.x) + x_try[e1.y].xyz * w.x;
        pb = x_try[e2.x].xyz * (1.0 - w.y) + x_try[e2.y].xyz * w.y;
    } else {
        pa = x_iter[e1.x].xyz * (1.0 - w.x) + x_iter[e1.y].xyz * w.x;
        pb = x_iter[e2.x].xyz * (1.0 - w.y) + x_iter[e2.y].xyz * w.y;
    }
    return pa - pb;
}

// Signed weight of vertex `v` in pair `p`'s stencil (0 if absent).
fn pair_weight_of(p: u32, v: u32) -> f32 {
    let m = pair_meta[p];
    let w = pair_wts[p];
    if (m.x == 0u) {
        if (v == m.y) { return 1.0; }
        let t = tris[m.z];
        if (v == t.x) { return -w.x; }
        if (v == t.y) { return -w.y; }
        if (v == t.z) { return -w.z; }
        return 0.0;
    }
    let e1 = edges[m.y];
    let e2 = edges[m.z];
    if (v == e1.x) { return 1.0 - w.x; }
    if (v == e1.y) { return w.x; }
    if (v == e2.x) { return -(1.0 - w.y); }
    if (v == e2.y) { return -w.y; }
    return 0.0;
}

// Reduced-mass κ of a pair (harmonic mean, same as the CPU oracle).
fn pair_kappa(p: u32) -> f32 {
    let m = pair_meta[p];
    var m1: f32;
    var m2: f32;
    var k1 = 0.0;
    var k2 = 0.0;
    if (m.x == 0u) {
        let t = tris[m.z];
        m1 = mass[m.y];
        m2 = (mass[t.x] + mass[t.y] + mass[t.z]) / 3.0;
        k1 = vertex_ksum[m.y];
        k2 = (vertex_ksum[t.x] + vertex_ksum[t.y] + vertex_ksum[t.z]) / 3.0;
    } else {
        let e1 = edges[m.y];
        let e2 = edges[m.z];
        m1 = 0.5 * (mass[e1.x] + mass[e1.y]);
        m2 = 0.5 * (mass[e2.x] + mass[e2.y]);
        k1 = 0.5 * (vertex_ksum[e1.x] + vertex_ksum[e1.y]);
        k2 = 0.5 * (vertex_ksum[e2.x] + vertex_ksum[e2.y]);
    }
    if (u.kappa_elastic != 0u) {
        m1 = m1 + u.h * u.h * k1;
        m2 = m2 + u.h * u.h * k2;
    }
    let s_red = 2.0 * m1 * m2 / max(m1 + m2, 1.0e-30);
    return u.kappa_scale * s_red / (u.h * u.h);
}

fn active_pair_count() -> u32 {
    return min(atomicLoad(&pair_count[0]), u.pair_capacity);
}

// Friction force profile g(s) and projected slope g′₊ (see the CPU
// oracle: stick ramp to μ_s·λ, blend down, kinetic plateau μ_k·λ).
fn friction_g(lambda: f32, s: f32, e: f32) -> vec2<f32> {
    let mu_s = u.mu_k * max(u.mu_s_scale, 1.0);
    if (s <= e) {
        let slope = mu_s * lambda / e;
        return vec2<f32>(slope * s, slope);
    }
    if (s <= 2.0 * e) {
        return vec2<f32>(mu_s * lambda + (u.mu_k - mu_s) * lambda * (s - e) / e, 0.0);
    }
    return vec2<f32>(u.mu_k * lambda, 0.0);
}

fn friction_energy_of(lambda: f32, s: f32, e: f32) -> f32 {
    let mu_s = u.mu_k * max(u.mu_s_scale, 1.0);
    let d_stick_e = mu_s * lambda * e / 2.0;
    if (s <= e) {
        return mu_s * lambda * s * s / (2.0 * e);
    }
    if (s <= 2.0 * e) {
        let t = s - e;
        return d_stick_e + mu_s * lambda * t + (u.mu_k - mu_s) * lambda * t * t / (2.0 * e);
    }
    return d_stick_e + mu_s * lambda * e + (u.mu_k - mu_s) * lambda * e / 2.0
        + u.mu_k * lambda * (s - 2.0 * e);
}

// Strain-limit barrier terms for spring s at strain: (b, b', b'') with
// gap = limit − strain, band = 20% of allowed over-strain, κ_s from
// the spring's own stiffness (elasticity-derived).
fn strain_barrier(sid: u32, strain: f32) -> vec3<f32> {
    if (u.strain_limit <= 0.0) { return vec3<f32>(0.0); }
    let gap = u.strain_limit - strain;
    let band = 0.2 * max(u.strain_limit - 1.0, 1.0e-6);
    if (gap >= band) { return vec3<f32>(0.0); }
    let kappa = u.strain_scale * spring_k[sid] * spring_rest[sid] * spring_rest[sid];
    let sgap = band - gap;
    let inv = 1.0 / (band * band);
    return vec3<f32>(
        kappa * sgap * sgap * sgap * inv / 3.0,
        -kappa * sgap * sgap * inv,
        2.0 * kappa * sgap * inv,
    );
}

fn tet_volume_at(ids: vec4<u32>, use_try: bool) -> f32 {
    var a: vec3<f32>; var b: vec3<f32>; var c: vec3<f32>; var d: vec3<f32>;
    if (use_try) {
        a = x_try[ids.x].xyz; b = x_try[ids.y].xyz;
        c = x_try[ids.z].xyz; d = x_try[ids.w].xyz;
    } else {
        a = x_iter[ids.x].xyz; b = x_iter[ids.y].xyz;
        c = x_iter[ids.z].xyz; d = x_iter[ids.w].xyz;
    }
    return dot(cross(b - a, c - a), d - a) / 6.0;
}

// Volume gradient of vertex-slot `slot` in tet `ids` at x_iter.
fn tet_grad_slot(ids: vec4<u32>, slot: u32) -> vec3<f32> {
    let pa = x_iter[ids.x].xyz;
    let pb = x_iter[ids.y].xyz;
    let pc = x_iter[ids.z].xyz;
    let pd = x_iter[ids.w].xyz;
    let gb = cross(pc - pa, pd - pa) / 6.0;
    let gc = cross(pd - pa, pb - pa) / 6.0;
    let gd = cross(pb - pa, pc - pa) / 6.0;
    if (slot == 1u) { return gb; }
    if (slot == 2u) { return gc; }
    if (slot == 3u) { return gd; }
    return -(gb + gc + gd);
}

// Rebuild the lagged friction rows from committed positions (called
// once per step, before the Newton loop).
@compute @workgroup_size(WG)
fn friction_lag(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = u.vertex_count * u.obstacle_count;
    if (idx >= total) { return; }
    let v = idx / u.obstacle_count;
    let o = idx % u.obstacle_count;
    var row = vec4<f32>(0.0);
    if (u.mu_k > 0.0 && pinned[v] == 0u) {
        let gd = obstacle_distance(o, x_pos[v].xyz);
        if (gd.w < u.dhat) {
            let s = max(u.dhat - gd.w, 0.0);
            // λ = h²·(−b′) with the SAME κ the barrier uses (a
            // mass-only λ under the inclusive barrier makes friction
            // proportionally weak — the CPU oracle's lesson).
            let lambda = u.h * u.h * vertex_kappa(v) * s * s / (u.dhat * u.dhat);
            row = vec4<f32>(x_pos[v].xyz, lambda);
        }
    }
    fric_lag[idx] = row;
}

// Elasticity-inclusive κ: scale·(m + h²·Σk)/h²; mass-only fallback.
fn vertex_kappa(i: u32) -> f32 {
    var m_eff = mass[i];
    if (u.kappa_elastic != 0u) {
        m_eff = m_eff + u.h * u.h * vertex_ksum[i];
    }
    return u.kappa_scale * m_eff / (u.h * u.h);
}

fn cell_hash(cx: i32, cy: i32, cz: i32) -> u32 {
    let h = (u32(cx) * 73856093u) ^ (u32(cy) * 19349663u) ^ (u32(cz) * 83492791u);
    return h & (u.grid_slots - 1u);
}

fn cell_of(p: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(floor(p / u.grid_cell));
}

@compute @workgroup_size(WG)
fn grid_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= 2u * u.grid_slots) { return; }
    atomicStore(&grid_count[gid.x], 0u);
}

fn grid_insert(base_slot: u32, lo: vec3<f32>, hi: vec3<f32>, id: u32) {
    let c0 = cell_of(lo);
    let c1 = cell_of(hi);
    for (var cx = c0.x; cx <= c1.x; cx = cx + 1) {
        for (var cy = c0.y; cy <= c1.y; cy = cy + 1) {
            for (var cz = c0.z; cz <= c1.z; cz = cz + 1) {
                let slot = base_slot + cell_hash(cx, cy, cz);
                let k = atomicAdd(&grid_count[slot], 1u);
                if (k < u.grid_cap) {
                    grid_items[slot * u.grid_cap + k] = id;
                } else {
                    atomicStore(&pair_count[1], 1u); // shared overflow flag
                }
            }
        }
    }
}

@compute @workgroup_size(WG)
fn grid_insert_tris(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= u.tri_count) { return; }
    let ids = tris[t];
    let a = x_iter[ids.x].xyz;
    let b = x_iter[ids.y].xyz;
    let c = x_iter[ids.z].xyz;
    grid_insert(0u, min(a, min(b, c)), max(a, max(b, c)), t);
}

@compute @workgroup_size(WG)
fn grid_insert_edges(@builtin(global_invocation_id) gid: vec3<u32>) {
    let e = gid.x;
    if (e >= u.edge_count) { return; }
    let ids = edges[e];
    let a = x_iter[ids.x].xyz;
    let b = x_iter[ids.y].xyz;
    grid_insert(u.grid_slots, min(a, b), max(a, b), e);
}

// Grid-pruned VT discovery: thread per vertex, probe the cells its
// 3·d̂ box overlaps, test only bucketed triangles.
@compute @workgroup_size(WG)
fn find_pairs_vt_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    if (v >= u.vertex_count) { return; }
    let p = x_iter[v].xyz;
    let m = vec3<f32>(3.0 * u.dhat);
    let c0 = cell_of(p - m);
    let c1 = cell_of(p + m);
    for (var cx = c0.x; cx <= c1.x; cx = cx + 1) {
        for (var cy = c0.y; cy <= c1.y; cy = cy + 1) {
            for (var cz = c0.z; cz <= c1.z; cz = cz + 1) {
                let slot = cell_hash(cx, cy, cz);
                let cnt = min(atomicLoad(&grid_count[slot]), u.grid_cap);
                for (var k = 0u; k < cnt; k = k + 1u) {
                    let t_idx = grid_items[slot * u.grid_cap + k];
                    let t = tris[t_idx];
                    if (v == t.x || v == t.y || v == t.z) { continue; }
                    // A tri can occupy several probed cells — accept it
                    // only from the FIRST probed cell that contains it
                    // (its AABB-lo cell clamped into the probe window)
                    // so duplicates never enter the pair list.
                    let a = x_iter[t.x].xyz;
                    let b = x_iter[t.y].xyz;
                    let c = x_iter[t.z].xyz;
                    let tlo = cell_of(min(a, min(b, c)));
                    let owner = vec3<i32>(
                        max(tlo.x, c0.x), max(tlo.y, c0.y), max(tlo.z, c0.z),
                    );
                    if (owner.x != cx || owner.y != cy || owner.z != cz) { continue; }
                    let w = tri_closest(p, a, b, c);
                    let cp = a * w.x + b * w.y + c * w.z;
                    if (length(p - cp) < 3.0 * u.dhat) {
                        let slot2 = atomicAdd(&pair_count[0], 1u);
                        if (slot2 < u.pair_capacity) {
                            pair_meta[slot2] = vec4<u32>(0u, v, t_idx, 0u);
                            pair_wts[slot2] = vec4<f32>(w, 0.0);
                        } else {
                            atomicStore(&pair_count[1], 1u);
                        }
                    }
                }
            }
        }
    }
}

// Grid-pruned EE discovery: thread per edge i, probe its own AABB+3d̂
// cells in the EDGE grid, test bucketed edges with j > i.
@compute @workgroup_size(WG)
fn find_pairs_ee_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.edge_count) { return; }
    let e1 = edges[i];
    let p1 = x_iter[e1.x].xyz;
    let q1 = x_iter[e1.y].xyz;
    let m = vec3<f32>(3.0 * u.dhat);
    let c0 = cell_of(min(p1, q1) - m);
    let c1 = cell_of(max(p1, q1) + m);
    for (var cx = c0.x; cx <= c1.x; cx = cx + 1) {
        for (var cy = c0.y; cy <= c1.y; cy = cy + 1) {
            for (var cz = c0.z; cz <= c1.z; cz = cz + 1) {
                let slot = u.grid_slots + cell_hash(cx, cy, cz);
                let cnt = min(atomicLoad(&grid_count[slot]), u.grid_cap);
                for (var k = 0u; k < cnt; k = k + 1u) {
                    let j = grid_items[slot * u.grid_cap + k];
                    if (j <= i) { continue; }
                    let e2 = edges[j];
                    if (e1.x == e2.x || e1.x == e2.y || e1.y == e2.x || e1.y == e2.y) {
                        continue;
                    }
                    let p2 = x_iter[e2.x].xyz;
                    let q2 = x_iter[e2.y].xyz;
                    let jlo = cell_of(min(p2, q2));
                    let owner = vec3<i32>(
                        max(jlo.x, c0.x), max(jlo.y, c0.y), max(jlo.z, c0.z),
                    );
                    if (owner.x != cx || owner.y != cy || owner.z != cz) { continue; }
                    let st = seg_closest(p1, q1, p2, q2);
                    let pa = p1 * (1.0 - st.x) + q1 * st.x;
                    let pb = p2 * (1.0 - st.y) + q2 * st.y;
                    if (length(pa - pb) < 3.0 * u.dhat) {
                        let slot2 = atomicAdd(&pair_count[0], 1u);
                        if (slot2 < u.pair_capacity) {
                            pair_meta[slot2] = vec4<u32>(1u, i, j, 0u);
                            pair_wts[slot2] = vec4<f32>(st.x, st.y, 0.0, 0.0);
                        } else {
                            atomicStore(&pair_count[1], 1u);
                        }
                    }
                }
            }
        }
    }
}

// Filter EXTERNAL VT candidates (mixed prim ids: verts 0..V, tris
// V..V+T) through the exact 3·d̂ test — replaces the hash probe when
// pair_source = 1. Vert-vert / tri-tri overlaps from the mixed BVH
// are skipped by the id-range check.
@compute @workgroup_size(WG)
fn filter_pairs_vt_cand(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (cand_meta[0] > arrayLength(&cand_vt_a)) {
        if (c == 0u) { atomicStore(&pair_count[1], 1u); }
        return;
    }
    if (c >= cand_meta[0]) { return; }
    var a = cand_vt_a[c];
    var b = cand_vt_b[c];
    if (a > b) { let t0 = a; a = b; b = t0; }
    if (a >= u.vertex_count || b < u.vertex_count) { return; }
    let v = a;
    let t_idx = b - u.vertex_count;
    let t = tris[t_idx];
    if (v == t.x || v == t.y || v == t.z) { return; }
    let pa = x_iter[t.x].xyz;
    let pb = x_iter[t.y].xyz;
    let pc = x_iter[t.z].xyz;
    let w = tri_closest(x_iter[v].xyz, pa, pb, pc);
    let cp = pa * w.x + pb * w.y + pc * w.z;
    if (length(x_iter[v].xyz - cp) < 3.0 * u.dhat) {
        let slot = atomicAdd(&pair_count[0], 1u);
        if (slot < u.pair_capacity) {
            pair_meta[slot] = vec4<u32>(0u, v, t_idx, 0u);
            pair_wts[slot] = vec4<f32>(w, 0.0);
        } else {
            atomicStore(&pair_count[1], 1u);
        }
    }
}

@compute @workgroup_size(WG)
fn filter_pairs_ee_cand(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (cand_meta[1] > arrayLength(&cand_ee_a)) {
        if (c == 0u) { atomicStore(&pair_count[1], 1u); }
        return;
    }
    if (c >= cand_meta[1]) { return; }
    var i = cand_ee_a[c];
    var j = cand_ee_b[c];
    if (i > j) { let t0 = i; i = j; j = t0; }
    if (i == j || j >= u.edge_count) { return; }
    let e1 = edges[i];
    let e2 = edges[j];
    if (e1.x == e2.x || e1.x == e2.y || e1.y == e2.x || e1.y == e2.y) { return; }
    let st = seg_closest(
        x_iter[e1.x].xyz, x_iter[e1.y].xyz, x_iter[e2.x].xyz, x_iter[e2.y].xyz,
    );
    let pa = x_iter[e1.x].xyz * (1.0 - st.x) + x_iter[e1.y].xyz * st.x;
    let pb = x_iter[e2.x].xyz * (1.0 - st.y) + x_iter[e2.y].xyz * st.y;
    if (length(pa - pb) < 3.0 * u.dhat) {
        let slot = atomicAdd(&pair_count[0], 1u);
        if (slot < u.pair_capacity) {
            pair_meta[slot] = vec4<u32>(1u, i, j, 0u);
            pair_wts[slot] = vec4<f32>(st.x, st.y, 0.0, 0.0);
        } else {
            atomicStore(&pair_count[1], 1u);
        }
    }
}



// Signed weight of vertex v in LAG pair p's stencil (mirror of
// pair_weight_of over the frozen arrays).
fn lag_weight_of(p: u32, v: u32) -> f32 {
    let m = lag_meta[p];
    let w = lag_wts[p];
    if (m.x == 0u) {
        if (v == m.y) { return 1.0; }
        let t = tris[m.z];
        if (v == t.x) { return -w.x; }
        if (v == t.y) { return -w.y; }
        if (v == t.z) { return -w.z; }
        return 0.0;
    }
    let e1 = edges[m.y];
    let e2 = edges[m.z];
    if (v == e1.x) { return 1.0 - w.x; }
    if (v == e1.y) { return w.x; }
    if (v == e2.x) { return -(1.0 - w.y); }
    if (v == e2.y) { return -w.y; }
    return 0.0;
}

// The four stencil vertex ids of lag pair p.
fn lag_stencil(p: u32) -> vec4<u32> {
    let m = lag_meta[p];
    if (m.x == 0u) {
        let t = tris[m.z];
        return vec4<u32>(m.y, t.x, t.y, t.z);
    }
    let e1 = edges[m.y];
    let e2 = edges[m.z];
    return vec4<u32>(e1.x, e1.y, e2.x, e2.y);
}

// Tangential relative slip of lag pair p at positions from
// x_try/x_iter: T·Σ wⱼ(xⱼ − anchorⱼ), n and w FROZEN.
fn lag_slip(p: u32, use_try: bool) -> vec3<f32> {
    let ids = lag_stencil(p);
    let nrm = lag_nl[p].xyz;
    var u_rel = vec3<f32>(0.0);
    for (var k = 0u; k < 4u; k = k + 1u) {
        let v = ids[k];
        let w = lag_weight_of(p, v);
        var xv: vec3<f32>;
        if (use_try) { xv = x_try[v].xyz; } else { xv = x_iter[v].xyz; }
        u_rel = u_rel + (xv - lag_anchors[p * 4u + k].xyz) * w;
    }
    return u_rel - nrm * dot(nrm, u_rel);
}

// Freeze the discovered pair set into the lag arrays (thread per
// discovered pair; run ONCE at step start, after discovery at the
// committed positions).
@compute @workgroup_size(WG)
fn pair_friction_lag(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= min(atomicLoad(&pair_count[0]), u.pair_capacity)) { return; }
    let diff = pair_diff(c, false);
    let dist = length(diff);
    if (dist >= u.dhat || dist <= 1.0e-9) { return; }
    let s = u.dhat - dist;
    let lambda = u.h * u.h * pair_kappa(c) * s * s / (u.dhat * u.dhat);
    let slot = atomicAdd(&pair_count[2], 1u);
    if (slot >= u.pair_capacity) {
        atomicStore(&pair_count[3], 1u);
        return;
    }
    lag_meta[slot] = pair_meta[c];
    lag_wts[slot] = pair_wts[c];
    lag_nl[slot] = vec4<f32>(diff / dist, lambda);
    let ids = lag_stencil(slot);
    for (var k = 0u; k < 4u; k = k + 1u) {
        lag_anchors[slot * 4u + k] = x_iter[ids[k]];
    }
}

@compute @workgroup_size(WG)
fn predict(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.vertex_count) { return; }
    let g = vec3<f32>(u.gravity_x, u.gravity_y, u.gravity_z);
    var xt = x_pos[i].xyz + vel[i].xyz * u.h + g * (u.h * u.h);
    if (pinned[i] != 0u) { xt = x_pos[i].xyz; }
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
        var f = h2 * spring_k[s] * (len - spring_rest[s]);
        if (u.strain_limit > 0.0) {
            // gap = limit − len/rest ⇒ the barrier ADDS tension.
            let sb = strain_barrier(s, len / spring_rest[s]);
            f = f - h2 * sb.y / spring_rest[s];
        }
        g = g - d / len * f;
    }
    let kappa = vertex_kappa(i);
    for (var o = 0u; o < u.obstacle_count; o = o + 1u) {
        let gd = obstacle_distance(o, xi);
        g = g + gd.xyz * (h2 * barrier_d1(gd.w, kappa));
    }
    // Self-pairs: race-free GATHER — each vertex scans the pair list
    // and accumulates its own signed-weight contribution. O(V·P), the
    // pre-LBVH tradeoff for determinism (no f32 atomics in WGSL).
    let np = active_pair_count();
    for (var p = 0u; p < np; p = p + 1u) {
        let w = pair_weight_of(p, i);
        if (w != 0.0) {
            let diff = pair_diff(p, false);
            let dist = max(length(diff), 1.0e-9);
            let db = barrier_d1(dist, pair_kappa(p));
            if (db != 0.0) {
                g = g + (diff / dist) * (h2 * db * w);
            }
        }
    }
    // Volumetric tets: dense scan, membership by id match.
    for (var t = 0u; t < u.tet_count; t = t + 1u) {
        let ids = tet_ids[t];
        var slot = 4u;
        if (i == ids.x) { slot = 0u; }
        if (i == ids.y) { slot = 1u; }
        if (i == ids.z) { slot = 2u; }
        if (i == ids.w) { slot = 3u; }
        if (slot < 4u) {
            let kv0 = tet_kv_v0[t];
            let dv = tet_volume_at(ids, false) - kv0.y;
            if (dv != 0.0) {
                g = g + tet_grad_slot(ids, slot) * (h2 * kv0.x * dv);
            }
        }
    }
    // Lagged friction rows for this vertex.
    if (u.mu_k > 0.0) {
        let ew = u.epsv * u.h;
        for (var o = 0u; o < u.obstacle_count; o = o + 1u) {
            let row = fric_lag[i * u.obstacle_count + o];
            if (row.w > 0.0) {
                let nrm = obstacle_distance(o, row.xyz).xyz;
                let uu = xi - row.xyz;
                let ut = uu - nrm * dot(nrm, uu);
                let sl = length(ut);
                if (sl > 1.0e-12) {
                    let gg = friction_g(row.w, sl, ew);
                    g = g + (ut / sl) * gg.x;
                }
            }
        }
    }
    if (u.mu_k > 0.0) {
        let ew2 = u.epsv * u.h;
        let nl = min(atomicLoad(&pair_count[2]), u.pair_capacity);
        for (var p = 0u; p < nl; p = p + 1u) {
            let w = lag_weight_of(p, i);
            if (w != 0.0) {
                let ut = lag_slip(p, false);
                let sl = length(ut);
                if (sl > 1.0e-12) {
                    let gg = friction_g(lag_nl[p].w, sl, ew2);
                    g = g + (ut / sl) * (gg.x * w);
                }
            }
        }
    }
    if (pinned[i] != 0u) { g = vec3<f32>(0.0); }
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
        var jdv = ((dv - dv_par) * t + dv_par) * (h2 * spring_k[s]);
        if (u.strain_limit > 0.0) {
            let sb = strain_barrier(s, len / spring_rest[s]);
            if (sb.z > 0.0) {
                jdv = jdv + dv_par * (h2 * sb.z / (spring_rest[s] * spring_rest[s]));
            }
        }
        y = y + jdv;
    }
    let kappa = vertex_kappa(i);
    for (var o = 0u; o < u.obstacle_count; o = o + 1u) {
        let gd = obstacle_distance(o, xi);
        let ddb = barrier_d2(gd.w, kappa);
        if (ddb > 0.0) {
            y = y + gd.xyz * (h2 * ddb * dot(gd.xyz, vi));
        }
    }
    // Rank-1 pair blocks, gathered: s = Σ_j w_j n·v_j needs the OTHER
    // stencil vertices' v — recomputed per vertex (deterministic).
    let np = active_pair_count();
    for (var p = 0u; p < np; p = p + 1u) {
        let wi = pair_weight_of(p, i);
        if (wi != 0.0) {
            let diff = pair_diff(p, false);
            let dist = max(length(diff), 1.0e-9);
            let ddb2 = barrier_d2(dist, pair_kappa(p));
            if (ddb2 > 0.0) {
                let n = diff / dist;
                let m = pair_meta[p];
                var s = 0.0;
                if (m.x == 0u) {
                    let t = tris[m.z];
                    let w = pair_wts[p];
                    s = dot(n, cg_d[m.y].xyz) - w.x * dot(n, cg_d[t.x].xyz)
                        - w.y * dot(n, cg_d[t.y].xyz) - w.z * dot(n, cg_d[t.z].xyz);
                } else {
                    let e1 = edges[m.y];
                    let e2 = edges[m.z];
                    let w = pair_wts[p];
                    s = (1.0 - w.x) * dot(n, cg_d[e1.x].xyz) + w.x * dot(n, cg_d[e1.y].xyz)
                        - (1.0 - w.y) * dot(n, cg_d[e2.x].xyz) - w.y * dot(n, cg_d[e2.y].xyz);
                }
                y = y + n * (h2 * ddb2 * wi * s);
            }
        }
    }
    // Tet Gauss blocks: k_v·g·gᵀ, gathered (needs the other slots' v).
    for (var t = 0u; t < u.tet_count; t = t + 1u) {
        let ids = tet_ids[t];
        var slot = 4u;
        if (i == ids.x) { slot = 0u; }
        if (i == ids.y) { slot = 1u; }
        if (i == ids.z) { slot = 2u; }
        if (i == ids.w) { slot = 3u; }
        if (slot < 4u) {
            let kv0 = tet_kv_v0[t];
            var s_acc = dot(tet_grad_slot(ids, 0u), cg_d[ids.x].xyz);
            s_acc = s_acc + dot(tet_grad_slot(ids, 1u), cg_d[ids.y].xyz);
            s_acc = s_acc + dot(tet_grad_slot(ids, 2u), cg_d[ids.z].xyz);
            s_acc = s_acc + dot(tet_grad_slot(ids, 3u), cg_d[ids.w].xyz);
            y = y + tet_grad_slot(ids, slot) * (h2 * kv0.x * s_acc);
        }
    }
    // Friction SPD blocks on the tangent plane.
    if (u.mu_k > 0.0) {
        let ew = u.epsv * u.h;
        for (var o = 0u; o < u.obstacle_count; o = o + 1u) {
            let row = fric_lag[i * u.obstacle_count + o];
            if (row.w > 0.0) {
                let nrm = obstacle_distance(o, row.xyz).xyz;
                let uu = xi - row.xyz;
                let ut = uu - nrm * dot(nrm, uu);
                let sl = length(ut);
                let vt = vi - nrm * dot(nrm, vi);
                if (sl > 1.0e-12) {
                    let gg = friction_g(row.w, sl, ew);
                    let uhat = ut / sl;
                    let v_par = uhat * dot(uhat, vt);
                    y = y + v_par * gg.y + (vt - v_par) * (gg.x / sl);
                } else {
                    let mu_s = u.mu_k * max(u.mu_s_scale, 1.0);
                    y = y + vt * (mu_s * row.w / ew);
                }
            }
        }
    }
    if (u.mu_k > 0.0) {
        let ew2 = u.epsv * u.h;
        let nl = min(atomicLoad(&pair_count[2]), u.pair_capacity);
        for (var p = 0u; p < nl; p = p + 1u) {
            let wi2 = lag_weight_of(p, i);
            if (wi2 != 0.0) {
                let nrm = lag_nl[p].xyz;
                let ids = lag_stencil(p);
                var vrel = vec3<f32>(0.0);
                for (var k = 0u; k < 4u; k = k + 1u) {
                    vrel = vrel + cg_d[ids[k]].xyz * lag_weight_of(p, ids[k]);
                }
                let vt = vrel - nrm * dot(nrm, vrel);
                let ut = lag_slip(p, false);
                let sl = length(ut);
                var hv: vec3<f32>;
                if (sl > 1.0e-12) {
                    let gg = friction_g(lag_nl[p].w, sl, ew2);
                    let uhat = ut / sl;
                    let v_par = uhat * dot(uhat, vt);
                    hv = v_par * gg.y + (vt - v_par) * (gg.x / sl);
                } else {
                    let mu_s = u.mu_k * max(u.mu_s_scale, 1.0);
                    hv = vt * (mu_s * lag_nl[p].w / ew2);
                }
                y = y + hv * wi2;
            }
        }
    }
    if (pinned[i] != 0u) { y = vi * mass[i]; }
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

// Warm start: keep the PREVIOUS Newton direction in p_dir and start
// CG from it — r = −grad − H·p_prev (H·p_prev arrives in cg_hd from a
// hessian_mul on the copied direction). Near-converged steps (resting
// contact) then need almost no refinement.
@compute @workgroup_size(WG)
fn copy_p_to_d(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.vertex_count) { return; }
    cg_d[i] = p_dir[i];
}

@compute @workgroup_size(WG)
fn cg_init_warm(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.vertex_count) { return; }
    let r = -grad[i].xyz - cg_hd[i].xyz;
    cg_r[i] = vec4<f32>(r, 0.0);
    cg_d[i] = vec4<f32>(r / mass[i], 0.0);
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
        // Global travel cap (with a mesh): no vertex moves more than
        // d̂ per accepted step — makes the 3·d̂ margin airtight.
        if (u.tri_count > 0u) {
            let speed0 = length(pi);
            if (speed0 > 0.0) {
                a_min = min(a_min, u.dhat / speed0);
            }
        }
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
    // HARD strain bound: truncate steps that would cross the limit.
    if (u.strain_limit > 0.0) {
        var sp = lid.x;
        loop {
            if (sp >= u.spring_count) { break; }
            let len = length(x_iter[spring_b[sp]].xyz - x_iter[spring_a[sp]].xyz);
            let gap = u.strain_limit - len / spring_rest[sp];
            let rate = (length(p_dir[spring_a[sp]].xyz) + length(p_dir[spring_b[sp]].xyz))
                / spring_rest[sp];
            if (rate > 0.0 && gap > 0.0) {
                a_min = min(a_min, 0.9 * gap / rate);
            }
            sp = sp + 256u;
        }
    }
    // Per-pair conservative TOI (summed max side speeds).
    let np = active_pair_count();
    var p = lid.x;
    loop {
        if (p >= np) { break; }
        let gap = length(pair_diff(p, false));
        let m = pair_meta[p];
        var closing: f32;
        if (m.x == 0u) {
            let t = tris[m.z];
            closing = length(p_dir[m.y].xyz)
                + max(length(p_dir[t.x].xyz),
                      max(length(p_dir[t.y].xyz), length(p_dir[t.z].xyz)));
        } else {
            let e1 = edges[m.y];
            let e2 = edges[m.z];
            closing = max(length(p_dir[e1.x].xyz), length(p_dir[e1.y].xyz))
                + max(length(p_dir[e2.x].xyz), length(p_dir[e2.y].xyz));
        }
        if (closing > 0.0) {
            a_min = min(a_min, 0.9 * gap / closing);
        }
        p = p + 256u;
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
    let kappa = vertex_kappa(i);
    for (var o = 0u; o < u.obstacle_count; o = o + 1u) {
        let gd = obstacle_distance(o, xi);
        e = e + u.h * u.h * barrier_energy(gd.w, kappa);
    }
    if (u.mu_k > 0.0) {
        let ew = u.epsv * u.h;
        for (var o = 0u; o < u.obstacle_count; o = o + 1u) {
            let row = fric_lag[i * u.obstacle_count + o];
            if (row.w > 0.0) {
                let nrm = obstacle_distance(o, row.xyz).xyz;
                let uu = xi - row.xyz;
                let ut = uu - nrm * dot(nrm, uu);
                e = e + friction_energy_of(row.w, length(ut), ew);
            }
        }
    }
    if (u.mu_k > 0.0) {
        let ew2 = u.epsv * u.h;
        let nl = min(atomicLoad(&pair_count[2]), u.pair_capacity);
        for (var p = 0u; p < nl; p = p + 1u) {
            if (lag_stencil(p).x == i) {
                e = e + friction_energy_of(lag_nl[p].w, length(lag_slip(p, true)), ew2);
            }
        }
    }
    // Tet energies, owned by the slot-0 vertex's thread.
    for (var t = 0u; t < u.tet_count; t = t + 1u) {
        let ids = tet_ids[t];
        if (ids.x == i) {
            let kv0 = tet_kv_v0[t];
            let dv = tet_volume_at(ids, true) - kv0.y;
            e = e + u.h * u.h * 0.5 * kv0.x * dv * dv;
        }
    }
    part[i] = e;
}

// Per-spring energy partials into part[vertex_count + s].
@compute @workgroup_size(WG)
fn energy_spring(@builtin(global_invocation_id) gid: vec3<u32>) {
    let s = gid.x;
    if (s >= u.spring_count) { return; }
    let d = x_try[spring_b[s]].xyz - x_try[spring_a[s]].xyz;
    let len = length(d);
    let c = len - spring_rest[s];
    var e = u.h * u.h * 0.5 * spring_k[s] * c * c;
    if (u.strain_limit > 0.0) {
        e = e + u.h * u.h * strain_barrier(s, len / spring_rest[s]).x;
    }
    part[u.vertex_count + s] = e;
}

// Per-pair barrier energies of x_try into part[NV + NS + p].
@compute @workgroup_size(WG)
fn energy_pairs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;
    if (p >= active_pair_count()) { return; }
    let dist = length(pair_diff(p, true));
    part[u.vertex_count + u.spring_count + p] =
        u.h * u.h * barrier_energy(dist, pair_kappa(p));
}

// Sum partials into scalars[7]; also min-gap of x_try into scalars[8]
// and ‖grad‖_∞ into scalars[9] (used by the host for convergence and
// the invariant WITHOUT extra passes).
@compute @workgroup_size(256)
fn reduce_energy_gap(@builtin(local_invocation_id) lid: vec3<u32>) {
    let total = u.vertex_count + u.spring_count + active_pair_count();
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
    // min gap over x_try: obstacles + frozen-topology self pairs.
    var g_min = 3.4e38;
    i = lid.x;
    loop {
        if (i >= u.vertex_count) { break; }
        for (var o = 0u; o < u.obstacle_count; o = o + 1u) {
            g_min = min(g_min, obstacle_distance(o, x_try[i].xyz).w);
        }
        i = i + 256u;
    }
    var pp = lid.x;
    let npg = active_pair_count();
    loop {
        if (pp >= npg) { break; }
        g_min = min(g_min, length(pair_diff(pp, true)));
        pp = pp + 256u;
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
    if (lid.x == 0u) {
        scalars[9] = red[0];
        scalars[10] = f32(atomicLoad(&pair_count[1])); // pair overflow
        scalars[11] = f32(atomicLoad(&pair_count[3])); // lag overflow
    }
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
    var v = (x_iter[i].xyz - x_pos[i].xyz) / u.h;
    if (pinned[i] != 0u) { v = vec3<f32>(0.0); }
    vel[i] = vec4<f32>(v, 0.0);
    x_pos[i] = x_iter[i];
}
