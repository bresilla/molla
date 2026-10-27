// Stage 24a — batched 3×3 small-matrix decompositions in WGSL.
//
// Three entry points, each running one matrix per thread:
//   * `svd_kernel`  — SVD via cyclic one-sided Jacobi rotations.
//   * `qr_kernel`   — QR via Givens rotations.
//   * `eig_kernel`  — symmetric eigendecomposition via cyclic Jacobi.
//
// Buffer layout (all f32):
//   group(0) binding(0) : input  matrices (row-major, 9 floats each).
//   group(0) binding(1) : output left  factor (U / Q / eigenvectors V).
//   group(0) binding(2) : output diag/triangular (σ / R / eigenvalues λ).
//   group(0) binding(3) : output right factor (Vᵀ for SVD; unused for
//                          QR + eig, written as identity to keep the
//                          three entry points share-compatible).
//   group(1) binding(0) : uniforms (count + which kernel).
//
// Convention matches the CPU side (`crates/molla/src/math/dense3.rs`):
//   M = U · diag(σ) · Vᵀ                   (svd)
//   M = Q · R                                (qr;  R upper-tri 9 floats)
//   M · v_i = λ_i · v_i                      (eig; V columns are v_i)
//
// The CPU comment in dense3.rs explains the algorithm choices in
// detail. The WGSL implementation is a literal port of the same
// arithmetic, kept in f32 because there's no f64 in WebGPU. Per-matrix
// convergence cap is 32 sweeps to bound execution time on bad inputs.

const JACOBI_EPS:        f32 = 1.0e-7;
const JACOBI_MAX_SWEEPS: u32 = 32u;

struct Uniforms {
    matrix_count: u32,
    _pad0:        u32,
    _pad1:        u32,
    _pad2:        u32,
};

@group(0) @binding(0) var<storage, read>       input_mats:  array<f32>;
@group(0) @binding(1) var<storage, read_write> out_left:    array<f32>;
@group(0) @binding(2) var<storage, read_write> out_middle:  array<f32>;
@group(0) @binding(3) var<storage, read_write> out_right:   array<f32>;
@group(1) @binding(0) var<uniform>             uniforms:    Uniforms;

// ── 3×3 mini-matrix helpers ────────────────────────────────────────
// Stored as 9 floats row-major: m[i*3 + j] is row i, col j.

fn load_mat(idx: u32) -> array<f32, 9> {
    let base = idx * 9u;
    var m: array<f32, 9>;
    for (var k: u32 = 0u; k < 9u; k = k + 1u) {
        m[k] = input_mats[base + k];
    }
    return m;
}

fn store(buf_idx: u32, idx: u32, m: array<f32, 9>) {
    // Three storage buffers — switch on which we want.
    let base = idx * 9u;
    if (buf_idx == 0u) {
        for (var k: u32 = 0u; k < 9u; k = k + 1u) {
            out_left[base + k] = m[k];
        }
    } else if (buf_idx == 1u) {
        for (var k: u32 = 0u; k < 9u; k = k + 1u) {
            out_middle[base + k] = m[k];
        }
    } else {
        for (var k: u32 = 0u; k < 9u; k = k + 1u) {
            out_right[base + k] = m[k];
        }
    }
}

fn store_vec(buf_idx: u32, idx: u32, v: vec3<f32>) {
    // Pack a vec3 into the diag-of-3-floats slot so the host can
    // read sigma / lambda back as 3 floats per matrix without a
    // separate buffer.
    let base = idx * 9u;
    if (buf_idx == 0u) {
        out_left[base + 0u] = v.x;
        out_left[base + 1u] = v.y;
        out_left[base + 2u] = v.z;
    } else if (buf_idx == 1u) {
        out_middle[base + 0u] = v.x;
        out_middle[base + 1u] = v.y;
        out_middle[base + 2u] = v.z;
    } else {
        out_right[base + 0u] = v.x;
        out_right[base + 1u] = v.y;
        out_right[base + 2u] = v.z;
    }
}

fn ident_mat() -> array<f32, 9> {
    return array<f32, 9>(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
}

// Matrix accessors (row-major).
fn at(m: ptr<function, array<f32, 9>>, i: u32, j: u32) -> f32 {
    return (*m)[i * 3u + j];
}
fn put(m: ptr<function, array<f32, 9>>, i: u32, j: u32, v: f32) {
    (*m)[i * 3u + j] = v;
}

fn col_of(m: ptr<function, array<f32, 9>>, j: u32) -> vec3<f32> {
    return vec3<f32>(at(m, 0u, j), at(m, 1u, j), at(m, 2u, j));
}
fn set_col(m: ptr<function, array<f32, 9>>, j: u32, v: vec3<f32>) {
    put(m, 0u, j, v.x);
    put(m, 1u, j, v.y);
    put(m, 2u, j, v.z);
}

fn mat_mul(a: ptr<function, array<f32, 9>>, b: ptr<function, array<f32, 9>>) -> array<f32, 9> {
    var out: array<f32, 9>;
    for (var i: u32 = 0u; i < 3u; i = i + 1u) {
        for (var j: u32 = 0u; j < 3u; j = j + 1u) {
            var s: f32 = 0.0;
            for (var k: u32 = 0u; k < 3u; k = k + 1u) {
                s = s + at(a, i, k) * at(b, k, j);
            }
            out[i * 3u + j] = s;
        }
    }
    return out;
}

fn transpose_mat(m: ptr<function, array<f32, 9>>) -> array<f32, 9> {
    var out: array<f32, 9>;
    for (var i: u32 = 0u; i < 3u; i = i + 1u) {
        for (var j: u32 = 0u; j < 3u; j = j + 1u) {
            out[i * 3u + j] = at(m, j, i);
        }
    }
    return out;
}

// ── SVD via cyclic one-sided Jacobi ────────────────────────────────

fn jacobi_pair(b: ptr<function, array<f32, 9>>, v: ptr<function, array<f32, 9>>, p: u32, q: u32) -> f32 {
    let bp = col_of(b, p);
    let bq = col_of(b, q);
    let app = dot(bp, bp);
    let aqq = dot(bq, bq);
    let apq = dot(bp, bq);
    let off = apq * apq;
    let scale = sqrt(app) * sqrt(aqq) + JACOBI_EPS;
    if (abs(apq) < JACOBI_EPS * scale) {
        return off;
    }
    let tau = (aqq - app) / (2.0 * apq);
    var t: f32;
    if (tau >= 0.0) {
        t = 1.0 / (tau + sqrt(1.0 + tau * tau));
    } else {
        t = 1.0 / (tau - sqrt(1.0 + tau * tau));
    }
    let c = 1.0 / sqrt(1.0 + t * t);
    let s = t * c;
    let bp_new = c * bp - s * bq;
    let bq_new = s * bp + c * bq;
    set_col(b, p, bp_new);
    set_col(b, q, bq_new);
    let vp = col_of(v, p);
    let vq = col_of(v, q);
    set_col(v, p, c * vp - s * vq);
    set_col(v, q, s * vp + c * vq);
    return off;
}

@compute @workgroup_size(64)
fn svd_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.matrix_count) { return; }
    var b = load_mat(idx);
    var v = ident_mat();
    for (var sweep: u32 = 0u; sweep < JACOBI_MAX_SWEEPS; sweep = sweep + 1u) {
        var off: f32 = 0.0;
        off = off + jacobi_pair(&b, &v, 0u, 1u);
        off = off + jacobi_pair(&b, &v, 0u, 2u);
        off = off + jacobi_pair(&b, &v, 1u, 2u);
        if (off < JACOBI_EPS * JACOBI_EPS) {
            break;
        }
    }
    var u = ident_mat();
    var sigma: vec3<f32>;
    for (var i: u32 = 0u; i < 3u; i = i + 1u) {
        let bi = col_of(&b, i);
        let n = length(bi);
        sigma[i] = n;
        if (n > 0.0) {
            set_col(&u, i, bi / n);
        }
        // else: leave as identity column.
    }
    // Force det(U) > 0 by flipping the smallest-σ column if needed.
    let det = dot(col_of(&u, 0u), cross(col_of(&u, 1u), col_of(&u, 2u)));
    if (det < 0.0) {
        var k: u32 = 0u;
        if (sigma[1u] < sigma[k]) { k = 1u; }
        if (sigma[2u] < sigma[k]) { k = 2u; }
        let uk = col_of(&u, k);
        set_col(&u, k, -uk);
        let vk = col_of(&v, k);
        set_col(&v, k, -vk);
    }
    store(0u, idx, u);
    store_vec(1u, idx, sigma);
    store(2u, idx, v);
}

// ── QR via Givens rotations ────────────────────────────────────────

fn givens_zero(r: ptr<function, array<f32, 9>>, q: ptr<function, array<f32, 9>>, i: u32, j: u32) {
    let a = at(r, j, j);
    let b = at(r, i, j);
    if (abs(b) < 1.0e-30) { return; }
    let h = sqrt(a * a + b * b);
    let c = a / h;
    let s = b / h;
    for (var k: u32 = 0u; k < 3u; k = k + 1u) {
        let rjk = at(r, j, k);
        let rik = at(r, i, k);
        put(r, j, k,  c * rjk + s * rik);
        put(r, i, k, -s * rjk + c * rik);
    }
    for (var k: u32 = 0u; k < 3u; k = k + 1u) {
        let qkj = at(q, k, j);
        let qki = at(q, k, i);
        put(q, k, j,  c * qkj + s * qki);
        put(q, k, i, -s * qkj + c * qki);
    }
}

@compute @workgroup_size(64)
fn qr_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.matrix_count) { return; }
    var r = load_mat(idx);
    var q = ident_mat();
    givens_zero(&r, &q, 1u, 0u);
    givens_zero(&r, &q, 2u, 0u);
    givens_zero(&r, &q, 2u, 1u);
    // Tidy sub-diagonal.
    put(&r, 1u, 0u, 0.0);
    put(&r, 2u, 0u, 0.0);
    put(&r, 2u, 1u, 0.0);
    store(0u, idx, q);
    store(1u, idx, r);
    var ident = ident_mat();
    store(2u, idx, ident);
}

// ── Symmetric eigendecomposition via cyclic Jacobi sweeps ─────────

fn eig_jacobi_pair(a: ptr<function, array<f32, 9>>, v: ptr<function, array<f32, 9>>, p: u32, q: u32) -> f32 {
    let apq = at(a, p, q);
    let off = apq * apq;
    if (abs(apq) < JACOBI_EPS) {
        return off;
    }
    let app = at(a, p, p);
    let aqq = at(a, q, q);
    let theta = (aqq - app) / (2.0 * apq);
    var t: f32;
    if (theta >= 0.0) {
        t = 1.0 / (theta + sqrt(1.0 + theta * theta));
    } else {
        t = 1.0 / (theta - sqrt(1.0 + theta * theta));
    }
    let c = 1.0 / sqrt(1.0 + t * t);
    let s = t * c;
    let app_new = app - t * apq;
    let aqq_new = aqq + t * apq;
    put(a, p, p, app_new);
    put(a, q, q, aqq_new);
    put(a, p, q, 0.0);
    put(a, q, p, 0.0);
    for (var r: u32 = 0u; r < 3u; r = r + 1u) {
        if (r == p || r == q) { continue; }
        let arp = at(a, r, p);
        let arq = at(a, r, q);
        let new_rp =  c * arp - s * arq;
        let new_rq =  s * arp + c * arq;
        put(a, r, p, new_rp);
        put(a, p, r, new_rp);
        put(a, r, q, new_rq);
        put(a, q, r, new_rq);
    }
    for (var r: u32 = 0u; r < 3u; r = r + 1u) {
        let vrp = at(v, r, p);
        let vrq = at(v, r, q);
        put(v, r, p, c * vrp - s * vrq);
        put(v, r, q, s * vrp + c * vrq);
    }
    return off;
}

@compute @workgroup_size(64)
fn eig_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.matrix_count) { return; }
    var raw = load_mat(idx);
    // Symmetrise input: a = (M + Mᵀ) / 2.
    var a: array<f32, 9>;
    a[0u] = raw[0u];
    a[4u] = raw[4u];
    a[8u] = raw[8u];
    let m01 = 0.5 * (raw[1u] + raw[3u]);
    let m02 = 0.5 * (raw[2u] + raw[6u]);
    let m12 = 0.5 * (raw[5u] + raw[7u]);
    a[1u] = m01;  a[3u] = m01;
    a[2u] = m02;  a[6u] = m02;
    a[5u] = m12;  a[7u] = m12;
    var v = ident_mat();
    for (var sweep: u32 = 0u; sweep < JACOBI_MAX_SWEEPS; sweep = sweep + 1u) {
        var off: f32 = 0.0;
        off = off + eig_jacobi_pair(&a, &v, 0u, 1u);
        off = off + eig_jacobi_pair(&a, &v, 0u, 2u);
        off = off + eig_jacobi_pair(&a, &v, 1u, 2u);
        if (off < JACOBI_EPS * JACOBI_EPS) { break; }
    }
    let lambda = vec3<f32>(at(&a, 0u, 0u), at(&a, 1u, 1u), at(&a, 2u, 2u));
    store(0u, idx, v);
    store_vec(1u, idx, lambda);
    var ident = ident_mat();
    store(2u, idx, ident);
}
