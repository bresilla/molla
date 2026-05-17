// Stage 38c — fused Adam parameter-update kernel.
//
// One thread per parameter:
//   m[i] = β₁·m[i] + (1−β₁)·g[i]
//   v[i] = β₂·v[i] + (1−β₂)·g[i]²
//   m̂   = m[i] / (1 − β₁^t)
//   v̂   = v[i] / (1 − β₂^t)
//   θ[i] -= lr · m̂ / (√v̂ + ε)
//   θ[i] -= lr · wd · θ_pre   if wd != 0          (decoupled, AdamW)
//
// `theta`, `m`, `v` are mutated in place. Reads `grad`. All buffers
// are `f32` (host-side `f64` precision lost at the buffer boundary —
// same convention as the rest of molla's GPU stack).
//
// Workgroup size 256 — matches `fem_vector_ops.wgsl`'s sizing for
// warp-aligned vector ops on NVIDIA / Intel / AMD desktop adapters.

const WG_SIZE: u32 = 256u;

struct AdamUniforms {
    n: u32,
    step: u32,           // 1-based current step index (set by host).
    _pad0: u32,
    _pad1: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    _pad2: f32,
    _pad3: f32,
    _pad4: f32,
};

@group(0) @binding(0) var<storage, read_write> adam_theta : array<f32>;
@group(0) @binding(1) var<storage, read>       adam_grad  : array<f32>;
@group(0) @binding(2) var<storage, read_write> adam_m     : array<f32>;
@group(0) @binding(3) var<storage, read_write> adam_v     : array<f32>;
@group(1) @binding(0) var<uniform>             adam_u     : AdamUniforms;

@compute @workgroup_size(WG_SIZE)
fn adam_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= adam_u.n) {
        return;
    }
    let g  = adam_grad[i];
    let m0 = adam_m[i];
    let v0 = adam_v[i];

    let m_new = adam_u.beta1 * m0 + (1.0 - adam_u.beta1) * g;
    let v_new = adam_u.beta2 * v0 + (1.0 - adam_u.beta2) * g * g;

    // Bias correction (`pow` is fine here — step counts are small,
    // and the alternative — host-side bc factors as a uniform —
    // means one more uniform field per dispatch).
    let t = f32(adam_u.step);
    let bc1 = max(1.0 - pow(adam_u.beta1, t), 1.0e-30);
    let bc2 = max(1.0 - pow(adam_u.beta2, t), 1.0e-30);
    let m_hat = m_new / bc1;
    let v_hat = v_new / bc2;

    let theta_pre = adam_theta[i];
    var theta = theta_pre - adam_u.lr * m_hat / (sqrt(v_hat) + adam_u.eps);
    if (adam_u.weight_decay != 0.0) {
        theta = theta - adam_u.lr * adam_u.weight_decay * theta_pre;
    }

    adam_m[i] = m_new;
    adam_v[i] = v_new;
    adam_theta[i] = theta;
}
