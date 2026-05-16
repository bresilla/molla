// Stage 38b — batched 3×3 adjoint SVD kernel.
//
// One thread per matrix. Given the forward decomposition
// `A = U · diag(σ) · Vᵀ` and the upstream gradients `adj_U`, `adj_σ`,
// `adj_V`, writes `adj_A = ∂J/∂A` for any downstream scalar loss `J`.
//
// Math mirrors `crates/molla/src/math/dense3.rs::adj_svd3` — see the
// Rust comments for the full derivation and the degenerate-σ
// regularisation (`F_ij = (σⱼ²−σᵢ²) / max((σⱼ²−σᵢ²)², ε²)`).
//
// Buffer layout (all f32, row-major 9 floats per matrix):
//   group(0) binding(0) : A       — forward input
//   group(0) binding(1) : U       — forward left singular vectors
//   group(0) binding(2) : sigma   — singular values (first 3 floats per slot)
//   group(0) binding(3) : V       — forward right singular vectors
//   group(0) binding(4) : adj_U   — upstream gradient w.r.t. U
//   group(0) binding(5) : adj_σ   — upstream gradient w.r.t. σ (first 3 floats)
//   group(0) binding(6) : adj_V   — upstream gradient w.r.t. V
//   group(0) binding(7) : adj_A   — output (this kernel writes it)
//   group(1) binding(0) : uniforms

const ADJ_EPS: f32 = 1.0e-6;

struct Uniforms {
    matrix_count: u32,
    _pad0:        u32,
    _pad1:        u32,
    _pad2:        u32,
};

@group(0) @binding(0) var<storage, read>       in_a:        array<f32>;
@group(0) @binding(1) var<storage, read>       in_u:        array<f32>;
@group(0) @binding(2) var<storage, read>       in_sigma:    array<f32>;
@group(0) @binding(3) var<storage, read>       in_v:        array<f32>;
@group(0) @binding(4) var<storage, read>       in_adj_u:    array<f32>;
@group(0) @binding(5) var<storage, read>       in_adj_sigma:array<f32>;
@group(0) @binding(6) var<storage, read>       in_adj_v:    array<f32>;
@group(0) @binding(7) var<storage, read_write> out_adj_a:   array<f32>;
@group(1) @binding(0) var<uniform>             uniforms:    Uniforms;

// ── 3×3 helpers (row-major 9-float arrays) ─────────────────────────

fn load_mat_from(buf: u32, idx: u32) -> array<f32, 9> {
    let base = idx * 9u;
    var m: array<f32, 9>;
    if (buf == 0u) {
        for (var k: u32 = 0u; k < 9u; k = k + 1u) { m[k] = in_a[base + k]; }
    } else if (buf == 1u) {
        for (var k: u32 = 0u; k < 9u; k = k + 1u) { m[k] = in_u[base + k]; }
    } else if (buf == 2u) {
        for (var k: u32 = 0u; k < 9u; k = k + 1u) { m[k] = in_v[base + k]; }
    } else if (buf == 3u) {
        for (var k: u32 = 0u; k < 9u; k = k + 1u) { m[k] = in_adj_u[base + k]; }
    } else {
        for (var k: u32 = 0u; k < 9u; k = k + 1u) { m[k] = in_adj_v[base + k]; }
    }
    return m;
}

fn at(m: ptr<function, array<f32, 9>>, i: u32, j: u32) -> f32 {
    return (*m)[i * 3u + j];
}
fn put(m: ptr<function, array<f32, 9>>, i: u32, j: u32, v: f32) {
    (*m)[i * 3u + j] = v;
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

fn mat_sub(a: ptr<function, array<f32, 9>>, b: ptr<function, array<f32, 9>>) -> array<f32, 9> {
    var out: array<f32, 9>;
    for (var k: u32 = 0u; k < 9u; k = k + 1u) {
        out[k] = (*a)[k] - (*b)[k];
    }
    return out;
}

fn mat_add3(a: array<f32, 9>, b: array<f32, 9>, c: array<f32, 9>) -> array<f32, 9> {
    var out: array<f32, 9>;
    for (var k: u32 = 0u; k < 9u; k = k + 1u) {
        out[k] = a[k] + b[k] + c[k];
    }
    return out;
}

fn cw_mul(a: ptr<function, array<f32, 9>>, b: ptr<function, array<f32, 9>>) -> array<f32, 9> {
    var out: array<f32, 9>;
    for (var k: u32 = 0u; k < 9u; k = k + 1u) {
        out[k] = (*a)[k] * (*b)[k];
    }
    return out;
}

fn diag_mat(d: vec3<f32>) -> array<f32, 9> {
    var out: array<f32, 9>;
    out[0] = d.x; out[1] = 0.0; out[2] = 0.0;
    out[3] = 0.0; out[4] = d.y; out[5] = 0.0;
    out[6] = 0.0; out[7] = 0.0; out[8] = d.z;
    return out;
}

fn safe_inv_sq_diff(a: f32, b: f32) -> f32 {
    let d = b * b - a * a;
    return d / (d * d + ADJ_EPS * ADJ_EPS);
}

@compute @workgroup_size(64)
fn adj_svd_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= uniforms.matrix_count) { return; }
    let base = idx * 9u;

    // Forward outputs.
    var u     = load_mat_from(1u, idx);
    var v     = load_mat_from(2u, idx);
    var adj_u = load_mat_from(3u, idx);
    var adj_v = load_mat_from(4u, idx);
    let sigma = vec3<f32>(in_sigma[base + 0u], in_sigma[base + 1u], in_sigma[base + 2u]);
    let asig  = vec3<f32>(
        in_adj_sigma[base + 0u],
        in_adj_sigma[base + 1u],
        in_adj_sigma[base + 2u],
    );

    // Build F (anti-symmetric, with regularised reciprocals).
    let f01 = safe_inv_sq_diff(sigma.x, sigma.y);
    let f02 = safe_inv_sq_diff(sigma.x, sigma.z);
    let f12 = safe_inv_sq_diff(sigma.y, sigma.z);
    var f: array<f32, 9>;
    f[0] =  0.0; f[1] =  f01; f[2] =  f02;
    f[3] = -f01; f[4] =  0.0; f[5] =  f12;
    f[6] = -f02; f[7] = -f12; f[8] =  0.0;

    var sigma_diag     = diag_mat(sigma);
    var adj_sigma_diag = diag_mat(asig);

    var ut = transpose_mat(&u);
    var vt = transpose_mat(&v);

    // 1) σ contribution: U · diag(adj_σ) · Vᵀ.
    var tmp_a = mat_mul(&u, &adj_sigma_diag);
    var sigma_term = mat_mul(&tmp_a, &vt);

    // 2) U-rotation contribution.
    //    Ω_U = Uᵀ·adj_U − adj_Uᵀ·U
    var ut_adju = mat_mul(&ut, &adj_u);
    var adju_t  = transpose_mat(&adj_u);
    var adjut_u = mat_mul(&adju_t, &u);
    var omega_u = mat_sub(&ut_adju, &adjut_u);
    var skew_u  = cw_mul(&f, &omega_u);
    var block_u = mat_mul(&skew_u, &sigma_diag);
    var tmp_b   = mat_mul(&u, &block_u);
    var u_term  = mat_mul(&tmp_b, &vt);

    // 3) V-rotation contribution.
    //    Ω_V = Vᵀ·adj_V − adj_Vᵀ·V
    var vt_adjv = mat_mul(&vt, &adj_v);
    var adjv_t  = transpose_mat(&adj_v);
    var adjvt_v = mat_mul(&adjv_t, &v);
    var omega_v = mat_sub(&vt_adjv, &adjvt_v);
    var skew_v  = cw_mul(&f, &omega_v);
    var block_v = mat_mul(&sigma_diag, &skew_v);
    var tmp_c   = mat_mul(&u, &block_v);
    var v_term  = mat_mul(&tmp_c, &vt);

    let adj_a = mat_add3(sigma_term, u_term, v_term);

    for (var k: u32 = 0u; k < 9u; k = k + 1u) {
        out_adj_a[base + k] = adj_a[k];
    }
}
