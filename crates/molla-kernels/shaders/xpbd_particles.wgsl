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
    /// Repurposed as `shape_count` for the `particle_vs_shapes` pass.
    color_index:       u32,
    /// Number of valid uniform-write slots in this dispatch — equals
    /// the colour's spring count when running a colour pass, else
    /// `particle_count` for predict / floor / velocity dispatches.
    work_count:        u32,
    /// Number of buckets in the self-collision spatial hash. The
    /// kernel hashes (cell.x, cell.y, cell.z) modulo this number.
    hash_buckets:      u32,
    /// Inverse cell size used by the hash. Cell size = 2 × particle
    /// radius works well for cloth where every particle has the same
    /// radius; multiply by a small slack (1.1×) so the 3×3×3 stencil
    /// captures all overlapping pairs.
    cell_inv_size:     f32,
    _pad0:             f32,
    _pad1:             f32,
    _pad2:             f32,
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

// ── Group 4: rigid shape collision data + self-collision hash ──────
struct Transform {
    position: vec4<f32>,
    rotation: vec4<f32>,
};
@group(4) @binding(0) var<storage, read>       shape_geom_kind:  array<u32>;
@group(4) @binding(1) var<storage, read>       shape_geom_data:  array<vec4<f32>>;
@group(4) @binding(2) var<storage, read>       shape_body:       array<i32>;
@group(4) @binding(3) var<storage, read>       shape_transform:  array<Transform>;
@group(4) @binding(4) var<storage, read>       body_q:           array<Transform>;
/// Spatial hash table: per-bucket count + fixed-capacity index list.
/// Layout: `hash_indices[bucket * MAX_PER_CELL + slot]`. Hash
/// collisions just produce extra distance checks (false positives),
/// not incorrect results, so a relatively small bucket count is fine.
@group(4) @binding(5) var<storage, read_write> hash_count:       array<atomic<u32>>;
@group(4) @binding(6) var<storage, read_write> hash_indices:     array<u32>;

const KIND_SPHERE:   u32 = 0u;
const KIND_PLANE:    u32 = 1u;
const KIND_BOX:      u32 = 2u;
const KIND_CAPSULE:  u32 = 3u;
const KIND_CYLINDER: u32 = 4u;
const KIND_SKIP:     u32 = 0xffu;
const STATIC_BODY:   i32 = -1;

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
// Quaternion + transform helpers (mirrors raycast.wgsl).
// ─────────────────────────────────────────────────────────────────
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = q.xyz;
    let t = 2.0 * cross(qv, v);
    return v + q.w * t + cross(qv, t);
}
fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}
fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    );
}
fn xf_compose(parent: Transform, local: Transform) -> Transform {
    var out: Transform;
    out.position = vec4<f32>(parent.position.xyz + quat_rotate(parent.rotation, local.position.xyz), 0.0);
    out.rotation = quat_mul(parent.rotation, local.rotation);
    return out;
}
fn xf_inverse(xf: Transform) -> Transform {
    let inv_rot = quat_conj(xf.rotation);
    var out: Transform;
    out.rotation = inv_rot;
    out.position = vec4<f32>(-quat_rotate(inv_rot, xf.position.xyz), 0.0);
    return out;
}
fn world_xf_of(shape: u32) -> Transform {
    let body = shape_body[shape];
    let local = shape_transform[shape];
    if (body == STATIC_BODY) {
        return local;
    }
    return xf_compose(body_q[u32(body)], local);
}

// Particle-vs-shape SDF + outward normal in WORLD frame. Returns
// `(world_normal_xyz, signed_distance)` packed into a vec4 (xyz =
// normal, w = sd). `sd > 0` outside, `sd < 0` inside.
fn sd_sphere(p_world: vec3<f32>, xf: Transform, radius: f32) -> vec4<f32> {
    let d = p_world - xf.position.xyz;
    let len = length(d);
    if (len < 1e-9) {
        return vec4<f32>(0.0, 1.0, 0.0, -radius);
    }
    return vec4<f32>(d / len, len - radius);
}

fn sd_plane(p_world: vec3<f32>, xf: Transform, n_local: vec3<f32>) -> vec4<f32> {
    let n_world = normalize(quat_rotate(xf.rotation, n_local));
    let sd = dot(p_world - xf.position.xyz, n_world);
    return vec4<f32>(n_world, sd);
}

fn sd_box(p_world: vec3<f32>, xf: Transform, half: vec3<f32>) -> vec4<f32> {
    let inv = xf_inverse(xf);
    let p_local = inv.position.xyz + quat_rotate(inv.rotation, p_world);
    let q = abs(p_local) - half;
    let outside = max(q, vec3<f32>(0.0));
    let outside_len = length(outside);
    let inside_d = min(max(q.x, max(q.y, q.z)), 0.0);
    let sd = outside_len + inside_d;

    var n_local: vec3<f32>;
    if (outside_len > 1e-9) {
        // Outside: normal points along the positive components of `q`.
        n_local = sign(p_local) * (outside / outside_len);
    } else {
        // Inside: pick the axis whose face is closest. `q` is negative
        // here, so we want max(q) which is the least-negative.
        var n = vec3<f32>(0.0, 1.0, 0.0);
        if (q.x >= q.y && q.x >= q.z) {
            n = vec3<f32>(sign(p_local.x), 0.0, 0.0);
        } else if (q.y >= q.z) {
            n = vec3<f32>(0.0, sign(p_local.y), 0.0);
        } else {
            n = vec3<f32>(0.0, 0.0, sign(p_local.z));
        }
        n_local = n;
    }
    let n_world = normalize(quat_rotate(xf.rotation, n_local));
    return vec4<f32>(n_world, sd);
}

fn sd_capsule(p_world: vec3<f32>, xf: Transform, half_h: f32, radius: f32) -> vec4<f32> {
    // Capsule axis = +Y in shape frame, segment from (0,-h,0) to (0,+h,0).
    let inv = xf_inverse(xf);
    let p_local = inv.position.xyz + quat_rotate(inv.rotation, p_world);
    var seg_t = clamp(p_local.y, -half_h, half_h);
    let seg_pt = vec3<f32>(0.0, seg_t, 0.0);
    let d_local = p_local - seg_pt;
    let len = length(d_local);
    var n_local: vec3<f32>;
    if (len > 1e-9) {
        n_local = d_local / len;
    } else {
        n_local = vec3<f32>(1.0, 0.0, 0.0);
    }
    let n_world = normalize(quat_rotate(xf.rotation, n_local));
    return vec4<f32>(n_world, len - radius);
}

fn sd_cylinder(p_world: vec3<f32>, xf: Transform, half_h: f32, radius: f32) -> vec4<f32> {
    // Finite cylinder, axis +Y, height 2*half_h, radius `radius`.
    let inv = xf_inverse(xf);
    let p_local = inv.position.xyz + quat_rotate(inv.rotation, p_world);
    let r_xz = length(vec2<f32>(p_local.x, p_local.z));
    let dx = r_xz - radius;
    let dy = abs(p_local.y) - half_h;
    let outside = vec2<f32>(max(dx, 0.0), max(dy, 0.0));
    let outside_len = length(outside);
    let inside_d = min(max(dx, dy), 0.0);
    let sd = outside_len + inside_d;
    var n_local: vec3<f32>;
    if (r_xz > 1e-9) {
        let n_radial = vec3<f32>(p_local.x / r_xz, 0.0, p_local.z / r_xz);
        if (outside_len > 1e-9) {
            // Outside surface: pick the dominant axis of `outside`.
            if (outside.x > outside.y) {
                n_local = n_radial;
            } else {
                n_local = vec3<f32>(0.0, sign(p_local.y), 0.0);
            }
        } else {
            // Inside: nearest face.
            if (dx > dy) {
                n_local = n_radial;
            } else {
                n_local = vec3<f32>(0.0, sign(p_local.y), 0.0);
            }
        }
    } else {
        // On the axis — pick the cap face direction.
        n_local = vec3<f32>(0.0, sign(p_local.y), 0.0);
        if (abs(p_local.y) < 1e-9) {
            n_local = vec3<f32>(1.0, 0.0, 0.0);
        }
    }
    let n_world = normalize(quat_rotate(xf.rotation, n_local));
    return vec4<f32>(n_world, sd);
}

// ─────────────────────────────────────────────────────────────────
// Particle-vs-rigid collision: project each particle out of every
// shape it penetrates. `u.work_count` = particle_count; `u.spring_count`
// is reused as `shape_count` so we don't need a new uniform field.
// ─────────────────────────────────────────────────────────────────
@compute @workgroup_size(64)
fn particle_vs_shapes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.work_count) { return; }
    if (particle_inv_mass[i] <= 0.0) { return; }

    let pr = particle_radius[i];
    var p = particle_q[i].xyz;
    let shape_count = u.color_index; // stuffed via `color_index` field
    for (var s: u32 = 0u; s < shape_count; s = s + 1u) {
        let kind = shape_geom_kind[s];
        if (kind == KIND_SKIP) { continue; }
        let xf = world_xf_of(s);
        let data = shape_geom_data[s].xyz;
        var nd: vec4<f32>;
        if (kind == KIND_SPHERE) {
            nd = sd_sphere(p, xf, data.x);
        } else if (kind == KIND_PLANE) {
            nd = sd_plane(p, xf, data);
        } else if (kind == KIND_BOX) {
            nd = sd_box(p, xf, data);
        } else if (kind == KIND_CAPSULE) {
            nd = sd_capsule(p, xf, data.x, data.y);
        } else if (kind == KIND_CYLINDER) {
            nd = sd_cylinder(p, xf, data.x, data.y);
        } else {
            continue;
        }
        let pen = pr - nd.w;
        if (pen > 0.0) {
            // Project particle along outward normal so it sits exactly
            // at distance `pr` outside the surface.
            p = p + nd.xyz * pen;
        }
    }
    particle_q[i] = vec4<f32>(p, 0.0);
}

// ─────────────────────────────────────────────────────────────────
// Spatial-hash self-collision. Three passes per substep:
//
//   hash_clear     — zero `hash_count` (one thread per bucket).
//   hash_insert    — atomically write the particle's index into its
//                    bucket. work_count = particle_count.
//   self_collide   — for each particle, scan the 3×3×3 cell stencil,
//                    accumulate Jacobi-style position corrections
//                    into `deltas[i]` / `counts[i]`.
//   strain_apply   — reused (last pass) to apply the averaged delta.
//
// `hash_buckets` and `cell_inv_size` come from `Uniforms`. Bucket
// capacity is the constant `MAX_PER_CELL` below.
// ─────────────────────────────────────────────────────────────────
const MAX_PER_CELL: u32 = 12u;
const HASH_P1: u32 = 73856093u;
const HASH_P2: u32 = 19349663u;
const HASH_P3: u32 = 83492791u;

fn cell_of(p: vec3<f32>) -> vec3<i32> {
    return vec3<i32>(floor(p * u.cell_inv_size));
}

fn cell_hash(c: vec3<i32>) -> u32 {
    let h = (u32(c.x) * HASH_P1) ^ (u32(c.y) * HASH_P2) ^ (u32(c.z) * HASH_P3);
    return h % u.hash_buckets;
}

@compute @workgroup_size(64)
fn hash_clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.work_count) { return; }
    atomicStore(&hash_count[i], 0u);
}

@compute @workgroup_size(64)
fn hash_insert(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.work_count) { return; }
    let p = particle_q[i].xyz;
    let bucket = cell_hash(cell_of(p));
    let slot = atomicAdd(&hash_count[bucket], 1u);
    if (slot < MAX_PER_CELL) {
        hash_indices[bucket * MAX_PER_CELL + slot] = i;
    }
    // Slots beyond MAX_PER_CELL are silently dropped — for cloth
    // densities that means "extremely unlikely missed collision",
    // not a crash. Bump MAX_PER_CELL if a scene ever hits this.
}

@compute @workgroup_size(64)
fn self_collide(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.work_count) { return; }
    let inv_m_i = particle_inv_mass[i];
    if (inv_m_i <= 0.0) {
        deltas[i] = vec4<f32>(0.0);
        counts[i] = 0u;
        return;
    }
    let r_i = particle_radius[i];
    let pos_i = particle_q[i].xyz;
    let cell = cell_of(pos_i);
    var dp = vec3<f32>(0.0);
    var n_hits: u32 = 0u;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let nc = cell + vec3<i32>(dx, dy, dz);
                let bucket = cell_hash(nc);
                let raw = atomicLoad(&hash_count[bucket]);
                let cnt = min(raw, MAX_PER_CELL);
                for (var s: u32 = 0u; s < cnt; s = s + 1u) {
                    let j = hash_indices[bucket * MAX_PER_CELL + s];
                    if (j == i) { continue; }
                    let pos_j = particle_q[j].xyz;
                    let r_j = particle_radius[j];
                    let inv_m_j = particle_inv_mass[j];
                    let w_sum = inv_m_i + inv_m_j;
                    if (w_sum <= 0.0) { continue; }
                    let d = pos_j - pos_i;
                    let len = length(d);
                    let min_d = r_i + r_j;
                    if (len < min_d && len > 1e-9) {
                        // XPBD-style position projection: this thread
                        // only writes its own correction, so there's
                        // no race with neighbour threads (each writes
                        // its own `deltas[j]` slot in its own
                        // dispatch).
                        let n = d / len;
                        let pen = min_d - len;
                        // Particle i moves opposite to `d` by its
                        // mass-weighted share of the penetration.
                        dp = dp - n * (pen * inv_m_i / w_sum);
                        n_hits = n_hits + 1u;
                    }
                }
            }
        }
    }
    deltas[i] = vec4<f32>(dp, 0.0);
    counts[i] = n_hits;
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
