// Parallel particle XPBD step.
//
// Replaces the original single-thread `step_particles_kernel`, which
// hung the OS via Timeout-Detection-and-Recovery on cloth scenes
// (≈600 k sequential ops per substep). Now split into six kernels,
// each parallel over either particles or springs in a single colour:
//
//   predict             — one thread per particle.
//   solve_distance      — one thread per spring in the current colour.
//                         Dispatched K times per XPBD iter, where K
//                         is `spring_color_count`. No two threads in
//                         a colour write the same particle, so no
//                         atomics or races.
//   strain_zero_scratch — one thread per particle (zeros deltas/counts).
//   strain_accumulate   — one thread per spring in the current colour.
//   strain_apply        — one thread per particle (averaged correction).
//   floor               — one thread per particle (Y=0 plane).
//   velocity_recovery   — one thread per particle.
//
// The host (`XpbdParticlesKernel::dispatch`) drives the encoder,
// alternating colours and looping over iteration counts.

struct Uniforms {
    particle_count:    u32,
    spring_count:      u32,
    iterations:        u32,
    dt:                f32,
    linear_damping:    f32,
    /// Slot in `spring_color_offsets` for the colour the current
    /// `solve_distance` / `strain_accumulate` dispatch is processing.
    /// Number of springs in the colour = offsets[idx+1] - offsets[idx].
    color_index:       u32,
    /// Number of valid uniform-write slots in this dispatch — equals
    /// the colour's spring count when running a colour pass, else
    /// `particle_count` for predict / floor / velocity dispatches.
    work_count:        u32,
    _pad:              u32,
};

const MAX_STRETCH_DISABLED: f32 = 1.0;

// ── Group 0: read-only model + state-in ─────────────────────────────
@group(0) @binding(0) var<storage, read>      particle_q_in:    array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>      particle_qd_in:   array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>      particle_inv_mass: array<f32>;
@group(0) @binding(3) var<storage, read>      particle_radius:  array<f32>;
@group(0) @binding(4) var<storage, read>      particle_world:   array<i32>;
@group(0) @binding(5) var<storage, read>      gravity:          array<vec4<f32>>;

// ── Group 1: read-only spring topology + colouring ──────────────────
@group(1) @binding(0) var<storage, read>      spring_a:                array<u32>;
@group(1) @binding(1) var<storage, read>      spring_b:                array<u32>;
@group(1) @binding(2) var<storage, read>      spring_rest_length:      array<f32>;
@group(1) @binding(3) var<storage, read>      spring_compliance:       array<f32>;
@group(1) @binding(4) var<storage, read>      spring_max_stretch:      array<f32>;
@group(1) @binding(5) var<storage, read>      spring_color_offsets:    array<u32>;
@group(1) @binding(6) var<storage, read>      spring_indices_by_color: array<u32>;

// ── Group 2: read/write working state ───────────────────────────────
@group(2) @binding(0) var<storage, read_write> particle_q:  array<vec4<f32>>;
@group(2) @binding(1) var<storage, read_write> particle_qd: array<vec4<f32>>;
@group(2) @binding(2) var<storage, read_write> lambdas:     array<f32>;
@group(2) @binding(3) var<storage, read_write> deltas:      array<vec4<f32>>;
@group(2) @binding(4) var<storage, read_write> counts:      array<u32>;

// ── Group 3: per-dispatch uniforms ──────────────────────────────────
@group(3) @binding(0) var<uniform>             u: Uniforms;

// ─────────────────────────────────────────────────────────────────
// Predict: gravity + linear damping + integrate position by dt.
// ─────────────────────────────────────────────────────────────────
@compute @workgroup_size(64)
fn predict(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.work_count) { return; }

    let inv_m = particle_inv_mass[i];
    if (inv_m <= 0.0) {
        // Pinned: pass through.
        particle_q[i] = particle_q_in[i];
        particle_qd[i] = vec4<f32>(0.0);
        return;
    }
    let world = max(particle_world[i], 0);
    let g = gravity[u32(world)].xyz;
    var v = particle_qd_in[i].xyz + g * u.dt;
    if (u.linear_damping > 0.0) {
        v = v * max(0.0, 1.0 - u.linear_damping * u.dt);
    }
    let p = particle_q_in[i].xyz + v * u.dt;
    particle_q[i] = vec4<f32>(p, 0.0);
    particle_qd[i] = vec4<f32>(v, 0.0);
}

// ─────────────────────────────────────────────────────────────────
// Reset the per-spring λ accumulator at the start of each substep.
// ─────────────────────────────────────────────────────────────────
@compute @workgroup_size(64)
fn reset_lambdas(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.work_count) { return; }
    lambdas[i] = 0.0;
}

// ─────────────────────────────────────────────────────────────────
// XPBD distance constraint, one colour pass. Race-free under
// graph colouring: no two springs in `color_index` share a particle.
// ─────────────────────────────────────────────────────────────────
@compute @workgroup_size(64)
fn solve_distance(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= u.work_count) { return; }
    let base = spring_color_offsets[u.color_index];
    let c = spring_indices_by_color[base + slot];

    let a = spring_a[c];
    let b = spring_b[c];
    if (a >= u.particle_count || b >= u.particle_count) { return; }
    let wa = particle_inv_mass[a];
    let wb = particle_inv_mass[b];
    let w_sum = wa + wb;
    if (w_sum <= 0.0) { return; }

    let pa = particle_q[a].xyz;
    let pb = particle_q[b].xyz;
    let delta = pb - pa;
    let len = length(delta);
    if (len < 1e-9) { return; }

    let n_dir = delta / len;
    let c_val = len - spring_rest_length[c];
    let dt2 = u.dt * u.dt;
    let alpha = spring_compliance[c] / dt2;
    let dlambda = -(c_val + alpha * lambdas[c]) / (w_sum + alpha);
    lambdas[c] = lambdas[c] + dlambda;
    let dp = n_dir * dlambda;
    particle_q[a] = vec4<f32>(pa - dp * wa, 0.0);
    particle_q[b] = vec4<f32>(pb + dp * wb, 0.0);
}

// ─────────────────────────────────────────────────────────────────
// Strain limit: zero per-particle Jacobi scratch (deltas + counts).
// ─────────────────────────────────────────────────────────────────
@compute @workgroup_size(64)
fn strain_zero(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.work_count) { return; }
    deltas[i] = vec4<f32>(0.0);
    counts[i] = 0u;
}

// ─────────────────────────────────────────────────────────────────
// Strain limit: accumulate per-spring corrections into per-particle
// scratch (one colour at a time, race-free under graph colouring).
// ─────────────────────────────────────────────────────────────────
@compute @workgroup_size(64)
fn strain_accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= u.work_count) { return; }
    let base = spring_color_offsets[u.color_index];
    let c = spring_indices_by_color[base + slot];

    let ms = spring_max_stretch[c];
    if (ms >= MAX_STRETCH_DISABLED) { return; }
    let a = spring_a[c];
    let b = spring_b[c];
    if (a >= u.particle_count || b >= u.particle_count) { return; }
    let wa = particle_inv_mass[a];
    let wb = particle_inv_mass[b];
    let w_sum = wa + wb;
    if (w_sum <= 0.0) { return; }

    let max_len = spring_rest_length[c] * (1.0 + ms);
    let pa = particle_q[a].xyz;
    let pb = particle_q[b].xyz;
    let delta = pb - pa;
    let len = length(delta);
    if (len > max_len && len > 1e-9) {
        let n_dir = delta / len;
        let excess = len - max_len;
        let dp = n_dir * (excess / w_sum);
        deltas[a] = vec4<f32>(deltas[a].xyz + dp * wa, 0.0);
        deltas[b] = vec4<f32>(deltas[b].xyz - dp * wb, 0.0);
        counts[a] = counts[a] + 1u;
        counts[b] = counts[b] + 1u;
    }
}

// ─────────────────────────────────────────────────────────────────
// Strain limit: apply averaged Jacobi correction to particles.
// ─────────────────────────────────────────────────────────────────
@compute @workgroup_size(64)
fn strain_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.work_count) { return; }
    if (particle_inv_mass[i] <= 0.0 || counts[i] == 0u) { return; }
    let avg = deltas[i].xyz / f32(counts[i]);
    particle_q[i] = vec4<f32>(particle_q[i].xyz + avg, 0.0);
}

// ─────────────────────────────────────────────────────────────────
// Y=0 plane floor collision.
// ─────────────────────────────────────────────────────────────────
@compute @workgroup_size(64)
fn floor_clamp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.work_count) { return; }
    if (particle_inv_mass[i] <= 0.0) { return; }
    let r = particle_radius[i];
    if (particle_q[i].y < r) {
        particle_q[i] = vec4<f32>(particle_q[i].x, r, particle_q[i].z, 0.0);
    }
}

// ─────────────────────────────────────────────────────────────────
// Velocity recovery: v = (p_new - p_in) / dt for free particles,
// zero for pinned (and restore initial p so floor clamp doesn't
// drift the pin).
// ─────────────────────────────────────────────────────────────────
@compute @workgroup_size(64)
fn velocity_recovery(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.work_count) { return; }
    if (particle_inv_mass[i] <= 0.0) {
        particle_q[i] = particle_q_in[i];
        particle_qd[i] = vec4<f32>(0.0);
        return;
    }
    let inv_dt = select(0.0, 1.0 / u.dt, u.dt > 1e-9);
    let p_new = particle_q[i].xyz;
    let p_in = particle_q_in[i].xyz;
    particle_qd[i] = vec4<f32>((p_new - p_in) * inv_dt, 0.0);
}
