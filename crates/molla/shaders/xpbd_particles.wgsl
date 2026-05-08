// XPBD particle step — predict + distance constraint projection +
// strain limit + plane collision + velocity recovery, all in one
// single-threaded GPU dispatch. Replaces the host-side
// `apply_particles_post_gpu` shim with a true GPU kernel so the cloth
// scene runs end-to-end on the device.
//
// Algorithm: faithful 1:1 port of `solvers::xpbd::step_particles`.
// Single-threaded — particle XPBD's distance-constraint projection
// is inherently Gauss-Seidel sequential (with a per-spring λ
// accumulator), and the strain limit uses Jacobi-style accumulation
// with deltas/counts that read-modify-write per-particle scratch
// buffers. Doing this with a single thread avoids needing
// atomic_add<f32> (unsupported in WGSL) at the cost of GPU
// parallelism. The win is eliminating the CPU↔GPU round-trip on
// every substep.
//
// Performance: 36×36 cloth (1296 particles, ~5000 springs) at 20 XPBD
// iters + 80 strain iters = ~600k single-thread GPU ops, well under a
// millisecond on any compute-capable adapter. Future optimisation:
// split into parallel passes for predict / floor / recover and keep
// the constraint loops single-threaded.
//
// Workgroup size 1, single dispatch.

struct Uniforms {
    particle_count:    u32,
    spring_count:      u32,
    iterations:        u32,
    dt:                f32,
    linear_damping:    f32,
    _pad0:             f32,
    _pad1:             f32,
    _pad2:             f32,
};

const MAX_STRETCH_DISABLED: f32 = 1.0;

@group(0) @binding(0) var<storage, read> particle_q_in:    array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> particle_qd_in:   array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> particle_inv_mass: array<f32>;
@group(0) @binding(3) var<storage, read> particle_radius:  array<f32>;
@group(0) @binding(4) var<storage, read> particle_world:   array<i32>;
@group(0) @binding(5) var<storage, read> gravity:          array<vec4<f32>>;

@group(1) @binding(0) var<storage, read> spring_a:           array<u32>;
@group(1) @binding(1) var<storage, read> spring_b:           array<u32>;
@group(1) @binding(2) var<storage, read> spring_rest_length: array<f32>;
@group(1) @binding(3) var<storage, read> spring_compliance:  array<f32>;
@group(1) @binding(4) var<storage, read> spring_max_stretch: array<f32>;

@group(2) @binding(0) var<storage, read_write> particle_q:  array<vec4<f32>>;
@group(2) @binding(1) var<storage, read_write> particle_qd: array<vec4<f32>>;
/// Per-spring λ accumulator (size = spring_count). Caller zeroes
/// this once before each substep.
@group(2) @binding(2) var<storage, read_write> lambdas:     array<f32>;
/// Per-particle scratch deltas for the Jacobi strain pass.
@group(2) @binding(3) var<storage, read_write> deltas:      array<vec4<f32>>;
/// Per-particle violation count for the Jacobi average.
@group(2) @binding(4) var<storage, read_write> counts:      array<u32>;

@group(3) @binding(0) var<uniform> u: Uniforms;

@compute @workgroup_size(1)
fn step_particles_kernel() {
    if (u.particle_count == 0u) { return; }

    // ── 1. Predict (sequential per particle). ────────────────────
    for (var i: u32 = 0u; i < u.particle_count; i = i + 1u) {
        let inv_m = particle_inv_mass[i];
        if (inv_m <= 0.0) {
            // Pinned: pass through, zero velocity. Final write happens
            // in the velocity-recovery loop below.
            particle_q[i] = particle_q_in[i];
            particle_qd[i] = vec4<f32>(0.0);
            continue;
        }
        let world = max(particle_world[i], 0);
        let g = gravity[u32(world)].xyz;
        var v = particle_qd_in[i].xyz + g * u.dt;
        if (u.linear_damping > 0.0) {
            v = v * max(0.0, 1.0 - u.linear_damping * u.dt);
        }
        let p = particle_q_in[i].xyz + v * u.dt;
        particle_q[i] = vec4<f32>(p, 0.0);
        // Stash the predicted v in particle_qd so the strain limit
        // sees current values; final velocity is recomputed in step 4.
        particle_qd[i] = vec4<f32>(v, 0.0);
    }

    // ── 2. Distance constraint projection (XPBD λ accumulator). ──
    let dt2 = u.dt * u.dt;
    if (u.spring_count > 0u) {
        // Reset λ accumulator (caller pre-zeroed it; redundant but
        // cheap and matches the CPU semantics that always starts at 0
        // each substep).
        for (var c: u32 = 0u; c < u.spring_count; c = c + 1u) {
            lambdas[c] = 0.0;
        }
        for (var iter: u32 = 0u; iter < u.iterations; iter = iter + 1u) {
            for (var c: u32 = 0u; c < u.spring_count; c = c + 1u) {
                let a = spring_a[c];
                let b = spring_b[c];
                if (a >= u.particle_count || b >= u.particle_count) {
                    continue;
                }
                let wa = particle_inv_mass[a];
                let wb = particle_inv_mass[b];
                let w_sum = wa + wb;
                if (w_sum <= 0.0) {
                    continue;
                }
                let pb = particle_q[b].xyz;
                let pa = particle_q[a].xyz;
                let delta = pb - pa;
                let len = length(delta);
                if (len < 1e-9) {
                    continue;
                }
                let n_dir = delta / len;
                let c_val = len - spring_rest_length[c];
                let alpha = spring_compliance[c] / dt2;
                let dlambda = -(c_val + alpha * lambdas[c]) / (w_sum + alpha);
                lambdas[c] = lambdas[c] + dlambda;
                let dp = n_dir * dlambda;
                particle_q[a] = vec4<f32>(pa - dp * wa, 0.0);
                particle_q[b] = vec4<f32>(pb + dp * wb, 0.0);
            }
        }

        // ── 2b. Hard strain limit (Jacobi over multiple iters). ──
        let strain_max_iters = u.iterations * 4u;
        for (var iter: u32 = 0u; iter < strain_max_iters; iter = iter + 1u) {
            // Reset Jacobi scratch.
            for (var i: u32 = 0u; i < u.particle_count; i = i + 1u) {
                deltas[i] = vec4<f32>(0.0);
                counts[i] = 0u;
            }
            var violated: bool = false;
            for (var c: u32 = 0u; c < u.spring_count; c = c + 1u) {
                let ms = spring_max_stretch[c];
                if (ms >= MAX_STRETCH_DISABLED) {
                    continue;
                }
                let a = spring_a[c];
                let b = spring_b[c];
                if (a >= u.particle_count || b >= u.particle_count) {
                    continue;
                }
                let wa = particle_inv_mass[a];
                let wb = particle_inv_mass[b];
                let w_sum = wa + wb;
                if (w_sum <= 0.0) {
                    continue;
                }
                let max_len = spring_rest_length[c] * (1.0 + ms);
                let pb = particle_q[b].xyz;
                let pa = particle_q[a].xyz;
                let delta = pb - pa;
                let len = length(delta);
                if (len > max_len && len > 1e-9) {
                    violated = true;
                    let n_dir = delta / len;
                    let excess = len - max_len;
                    let dp = n_dir * (excess / w_sum);
                    deltas[a] = vec4<f32>(deltas[a].xyz + dp * wa, 0.0);
                    deltas[b] = vec4<f32>(deltas[b].xyz - dp * wb, 0.0);
                    counts[a] = counts[a] + 1u;
                    counts[b] = counts[b] + 1u;
                }
            }
            if (!violated) {
                break;
            }
            // Apply averaged deltas, skipping pinned particles.
            for (var i: u32 = 0u; i < u.particle_count; i = i + 1u) {
                if (particle_inv_mass[i] <= 0.0 || counts[i] == 0u) {
                    continue;
                }
                let avg = deltas[i].xyz / f32(counts[i]);
                particle_q[i] = vec4<f32>(particle_q[i].xyz + avg, 0.0);
            }
        }
    }

    // ── 3. Floor (Y=0 plane) collision. ──────────────────────────
    for (var i: u32 = 0u; i < u.particle_count; i = i + 1u) {
        if (particle_inv_mass[i] <= 0.0) {
            continue;
        }
        let r = particle_radius[i];
        if (particle_q[i].y < r) {
            particle_q[i] = vec4<f32>(particle_q[i].x, r, particle_q[i].z, 0.0);
        }
    }

    // ── 4. Velocity recovery: v = (p_new - p_in) / dt. ───────────
    let inv_dt = select(0.0, 1.0 / u.dt, u.dt > 1e-9);
    for (var i: u32 = 0u; i < u.particle_count; i = i + 1u) {
        if (particle_inv_mass[i] <= 0.0) {
            particle_q[i] = particle_q_in[i];
            particle_qd[i] = vec4<f32>(0.0);
            continue;
        }
        let p_new = particle_q[i].xyz;
        let p_in = particle_q_in[i].xyz;
        particle_qd[i] = vec4<f32>((p_new - p_in) * inv_dt, 0.0);
    }
}
