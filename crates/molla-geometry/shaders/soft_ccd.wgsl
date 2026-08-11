// GPU vertex-triangle continuous collision detection. Mirrors the CPU
// `vt_ccd` in `crates/molla-geometry/src/soft_ccd.rs`: the signed
// tetrahedral volume f(t) = ((B-A)×(C-A))·(V-A) is a cubic in t; find
// its earliest root in [0,1] (sign-change sweep + bisection) whose
// barycentric projection lies inside the triangle.
//
// One thread per query. Output: earliest TOI in [0,1], or -1 for no hit.

struct Uniforms {
    counts: vec4<u32>,   // (query_count, _, _, _)
};

// Per query: verts[2q]=V(t0), verts[2q+1]=V(t1);
//            tri0[3q..]=A,B,C at t0; tri1[3q..]=A,B,C at t1.
@group(0) @binding(0) var<storage, read> verts: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> tri0:  array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> tri1:  array<vec4<f32>>;

@group(1) @binding(0) var<storage, read_write> out_toi: array<f32>;

@group(2) @binding(0) var<uniform> u: Uniforms;

const BARY_SLACK: f32 = 1.0e-4;
const ROOT_EPS: f32 = 1.0e-7;
const MAX_BISECT: u32 = 60u;
const N: u32 = 20u;

fn f_eval(
    t: f32,
    v0: vec3<f32>, dv: vec3<f32>,
    a0: vec3<f32>, da: vec3<f32>,
    b0: vec3<f32>, db: vec3<f32>,
    c0: vec3<f32>, dc: vec3<f32>,
) -> f32 {
    let v = v0 + dv * t;
    let a = a0 + da * t;
    let b = b0 + db * t;
    let c = c0 + dc * t;
    return dot(cross(b - a, c - a), v - a);
}

// Barycentric inside-triangle test of V(t)'s projection.
fn check_inside(
    t: f32,
    v0: vec3<f32>, dv: vec3<f32>,
    a0: vec3<f32>, da: vec3<f32>,
    b0: vec3<f32>, db: vec3<f32>,
    c0: vec3<f32>, dc: vec3<f32>,
) -> bool {
    let v = v0 + dv * t;
    let a = a0 + da * t;
    let b = b0 + db * t;
    let c = c0 + dc * t;
    let ab = b - a;
    let ac = c - a;
    let av = v - a;
    let d00 = dot(ab, ab);
    let d01 = dot(ab, ac);
    let d11 = dot(ac, ac);
    let d20 = dot(av, ab);
    let d21 = dot(av, ac);
    let denom = d00 * d11 - d01 * d01;
    if (abs(denom) < 1.0e-20) {
        return true; // degenerate triangle — be conservative
    }
    let vb = (d11 * d20 - d01 * d21) / denom;
    let wb = (d00 * d21 - d01 * d20) / denom;
    let ub = 1.0 - vb - wb;
    return ub >= -BARY_SLACK && vb >= -BARY_SLACK && wb >= -BARY_SLACK;
}

// Bisection refiner over a sign-change bracket. Returns the root t.
fn bisect(
    t_lo_in: f32, f_lo_in: f32, t_hi_in: f32,
    v0: vec3<f32>, dv: vec3<f32>,
    a0: vec3<f32>, da: vec3<f32>,
    b0: vec3<f32>, db: vec3<f32>,
    c0: vec3<f32>, dc: vec3<f32>,
) -> f32 {
    var t_lo = t_lo_in;
    var f_lo = f_lo_in;
    var t_hi = t_hi_in;
    for (var i = 0u; i < MAX_BISECT; i = i + 1u) {
        let t_mid = 0.5 * (t_lo + t_hi);
        let f_mid = f_eval(t_mid, v0, dv, a0, da, b0, db, c0, dc);
        if (abs(f_mid) < ROOT_EPS) {
            return t_mid;
        }
        if ((f_lo >= 0.0) != (f_mid >= 0.0)) {
            t_hi = t_mid;
        } else {
            t_lo = t_mid;
            f_lo = f_mid;
        }
        if (abs(t_hi - t_lo) < ROOT_EPS) {
            return 0.5 * (t_lo + t_hi);
        }
    }
    return 0.5 * (t_lo + t_hi);
}

@compute @workgroup_size(64)
fn vt_ccd(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = gid.x;
    if (q >= u.counts.x) { return; }

    let v0 = verts[2u * q + 0u].xyz;
    let dv = verts[2u * q + 1u].xyz - v0;
    let a0 = tri0[3u * q + 0u].xyz;
    let b0 = tri0[3u * q + 1u].xyz;
    let c0 = tri0[3u * q + 2u].xyz;
    let da = tri1[3u * q + 0u].xyz - a0;
    let db = tri1[3u * q + 1u].xyz - b0;
    let dc = tri1[3u * q + 2u].xyz - c0;

    var result: f32 = -1.0;

    let f0 = f_eval(0.0, v0, dv, a0, da, b0, db, c0, dc);
    if (abs(f0) < ROOT_EPS && check_inside(0.0, v0, dv, a0, da, b0, db, c0, dc)) {
        out_toi[q] = 0.0;
        return;
    }

    var prev_t = 0.0;
    var prev_f = f0;
    for (var k = 1u; k <= N; k = k + 1u) {
        let t = f32(k) / f32(N);
        let curr_f = f_eval(t, v0, dv, a0, da, b0, db, c0, dc);
        if ((prev_f >= 0.0) != (curr_f >= 0.0)) {
            let root = bisect(prev_t, prev_f, t, v0, dv, a0, da, b0, db, c0, dc);
            if (check_inside(root, v0, dv, a0, da, b0, db, c0, dc)) {
                result = root;
                break;
            }
        }
        prev_t = t;
        prev_f = curr_f;
    }

    out_toi[q] = result;
}
