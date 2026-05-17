// Stage 24b — robust-loss family in WGSL.
//
// One entry point per loss; each thread evaluates ρ(r) and ρ'(r) for
// a single sample. Output buffer layout: 2 floats per input, packed as
// `[ρ_0, ρ'_0, ρ_1, ρ'_1, …]`. The Rust wrapper unpacks both halves
// into separate `Vec<f32>` slices for parity-checking against the CPU
// reference.
//
// Buffer bindings (shared across all entry points):
//   group(0) binding(0) : input residuals  r_i        (read-only).
//   group(0) binding(1) : output (ρ, ρ')   pairs      (read-write).
//   group(1) binding(0) : uniforms { sample_count, param0, param1, _pad }
//
// `param0` carries the loss-shape parameter (δ for Huber / pseudo-Huber,
// ε for smooth-L2, c for Tukey / Cauchy). `param1` is reserved for
// future two-parameter losses (none today).

const ZERO_TOL: f32 = 1.0e-6;

struct Uniforms {
    sample_count: u32,
    param0:       f32,
    param1:       f32,
    _pad:         u32,
};

@group(0) @binding(0) var<storage, read>       residuals: array<f32>;
@group(0) @binding(1) var<storage, read_write> out_pair:  array<f32>;
@group(1) @binding(0) var<uniform>             uniforms:  Uniforms;

fn write_pair(idx: u32, rho: f32, drho: f32) {
    let base = idx * 2u;
    out_pair[base + 0u] = rho;
    out_pair[base + 1u] = drho;
}

// ── L1 ────────────────────────────────────────────────────────────

fn l1_eval(r: f32) -> vec2<f32> {
    let v = abs(r);
    var g: f32;
    if (r > 0.0) {
        g = 1.0;
    } else if (r < 0.0) {
        g = -1.0;
    } else {
        g = 0.0;
    }
    return vec2<f32>(v, g);
}

@compute @workgroup_size(64)
fn l1_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.sample_count) { return; }
    let r = residuals[i];
    let p = l1_eval(r);
    write_pair(i, p.x, p.y);
}

// ── Smooth L2 ─────────────────────────────────────────────────────

fn smooth_l2_eval(r: f32, eps: f32) -> vec2<f32> {
    let s = sqrt(r * r + eps * eps);
    let rho = s - eps;
    let g = r / s;
    return vec2<f32>(rho, g);
}

@compute @workgroup_size(64)
fn smooth_l2_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.sample_count) { return; }
    let r = residuals[i];
    let p = smooth_l2_eval(r, uniforms.param0);
    write_pair(i, p.x, p.y);
}

// ── Huber ─────────────────────────────────────────────────────────

fn huber_eval(r: f32, delta: f32) -> vec2<f32> {
    let absr = abs(r);
    var rho: f32;
    var g: f32;
    if (absr <= delta) {
        rho = 0.5 * r * r;
        g = r;
    } else {
        rho = delta * (absr - 0.5 * delta);
        g = delta * sign(r);
    }
    return vec2<f32>(rho, g);
}

@compute @workgroup_size(64)
fn huber_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.sample_count) { return; }
    let r = residuals[i];
    let p = huber_eval(r, uniforms.param0);
    write_pair(i, p.x, p.y);
}

// ── Pseudo-Huber ──────────────────────────────────────────────────

fn pseudo_huber_eval(r: f32, delta: f32) -> vec2<f32> {
    let s = r / delta;
    let root = sqrt(1.0 + s * s);
    let rho = delta * delta * (root - 1.0);
    let g = r / root;
    return vec2<f32>(rho, g);
}

@compute @workgroup_size(64)
fn pseudo_huber_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.sample_count) { return; }
    let r = residuals[i];
    let p = pseudo_huber_eval(r, uniforms.param0);
    write_pair(i, p.x, p.y);
}

// ── Tukey biweight ────────────────────────────────────────────────

fn tukey_eval(r: f32, c: f32) -> vec2<f32> {
    let cap = c * c / 6.0;
    let absr = abs(r);
    if (absr >= c) {
        return vec2<f32>(cap, 0.0);
    }
    let s = r / c;
    let u = 1.0 - s * s;
    let rho = cap * (1.0 - u * u * u);
    let g = r * u * u;
    return vec2<f32>(rho, g);
}

@compute @workgroup_size(64)
fn tukey_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.sample_count) { return; }
    let r = residuals[i];
    let p = tukey_eval(r, uniforms.param0);
    write_pair(i, p.x, p.y);
}

// ── Cauchy ────────────────────────────────────────────────────────

fn cauchy_eval(r: f32, c: f32) -> vec2<f32> {
    let s = r / c;
    let denom = 1.0 + s * s;
    let rho = 0.5 * c * c * log(denom);
    let g = r / denom;
    return vec2<f32>(rho, g);
}

@compute @workgroup_size(64)
fn cauchy_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.sample_count) { return; }
    let r = residuals[i];
    let p = cauchy_eval(r, uniforms.param0);
    write_pair(i, p.x, p.y);
}
