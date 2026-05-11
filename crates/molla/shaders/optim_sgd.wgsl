// Stage 38c — fused SGD parameter-update kernel.
//
// One thread per parameter:
//   g̃   = g[i] + wd · θ[i]
//   b̂   = μ·buf[i] + (1−damp)·g̃            if step > 0
//       = g̃                                otherwise (first step)
//   g_eff = g̃ + μ·b̂                         if Nesterov
//         = b̂                              otherwise
//   θ[i] -= lr · g_eff
//   buf[i] = b̂
//
// `theta` and `buf` are mutated in place. `grad` is read-only.
// If `momentum == 0`, the buffer is bypassed (no read/write); the
// update collapses to plain SGD.

const WG_SIZE: u32 = 256u;

struct SgdUniforms {
    n: u32,
    nesterov: u32,       // bool encoded as 0 / 1.
    step: u32,           // 0-based step index BEFORE the update.
    _pad0: u32,
    lr: f32,
    momentum: f32,
    dampening: f32,
    weight_decay: f32,
};

@group(0) @binding(0) var<storage, read_write> sgd_theta : array<f32>;
@group(0) @binding(1) var<storage, read>       sgd_grad  : array<f32>;
@group(0) @binding(2) var<storage, read_write> sgd_buf   : array<f32>;
@group(1) @binding(0) var<uniform>             sgd_u     : SgdUniforms;

@compute @workgroup_size(WG_SIZE)
fn sgd_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= sgd_u.n) {
        return;
    }
    let theta_pre = sgd_theta[i];
    var g = sgd_grad[i];
    if (sgd_u.weight_decay != 0.0) {
        g = g + sgd_u.weight_decay * theta_pre;
    }

    var g_eff: f32 = g;
    if (sgd_u.momentum != 0.0) {
        var b_new: f32;
        if (sgd_u.step == 0u) {
            b_new = g;
        } else {
            b_new = sgd_u.momentum * sgd_buf[i] + (1.0 - sgd_u.dampening) * g;
        }
        sgd_buf[i] = b_new;
        if (sgd_u.nesterov != 0u) {
            g_eff = g + sgd_u.momentum * b_new;
        } else {
            g_eff = b_new;
        }
    }

    sgd_theta[i] = theta_pre - sgd_u.lr * g_eff;
}
