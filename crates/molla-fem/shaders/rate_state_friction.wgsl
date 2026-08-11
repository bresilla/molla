// Batched Dieterich-Ruina rate-and-state friction advance. Mirrors the
// CPU `advance_rate_state_friction` in `crates/molla-fem/src/signorini.rs`:
// one explicit Euler step of the aging law dθ/dt = 1 − vθ/dc, then the
// steady coefficient μ = μ0 + a·ln(v/v0) + b·ln(θ v0/dc). One thread per
// contact.

struct U {
    p0: vec4<f32>,   // (mu0, a, b, v0)
    p1: vec4<f32>,   // (dc, dt, count, _)
};

@group(0) @binding(0) var<storage, read> slip:     array<f32>;
@group(0) @binding(1) var<storage, read> theta_in: array<f32>;

@group(1) @binding(0) var<storage, read_write> theta_out: array<f32>;
@group(1) @binding(1) var<storage, read_write> mu_out:    array<f32>;

@group(2) @binding(0) var<uniform> u: U;

@compute @workgroup_size(64)
fn advance(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let count = u32(u.p1.z);
    if (i >= count) { return; }

    let mu0 = u.p0.x;
    let a = u.p0.y;
    let b = u.p0.z;
    let v0 = u.p0.w;
    let dc = u.p1.x;
    let dt = u.p1.y;

    let v = max(slip[i], 0.0);
    let theta_dot = 1.0 - v * theta_in[i] / dc;
    let theta = max(theta_in[i] + max(dt, 0.0) * theta_dot, 1.0e-30);

    let vc = max(v, v0 * 1.0e-12);
    let th = max(theta, dc / v0 * 1.0e-12);
    let mu = mu0 + a * log(vc / v0) + b * log(th * v0 / dc);

    theta_out[i] = theta;
    mu_out[i] = mu;
}
