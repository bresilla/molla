// Stage 44 v1 — differentiable 3D-Gaussian splat raytrace (FORWARD ONLY).
//
// One thread per pixel. Each thread walks the full splat array,
// keeps a fixed-size top-K mini-sort of the closest hits in
// front-to-back order, then α-composites them.
//
// This is the forward path; the reverse adjoint (for inverse
// rendering / pose recovery) is a separate stage. We deliberately
// keep the per-splat math identical to Newton's `warp_raytrace/
// gaussians.py::canonical_ray_max_kernel_response` so a future
// v2 adjoint can reuse the same canonical-frame chain rule.
//
// Inputs (storage):
//   splat_pos    : per-splat world-space mean (xyz; w padded)
//   splat_scale  : per-splat anisotropic stddev (xyz; w padded)
//   splat_rot    : per-splat unit quaternion (x, y, z, w)
//   splat_color  : per-splat diffuse RGB + reserved alpha slot
//   splat_opacity: per-splat bare opacity ∈ [0, 1]
//   ray_origin   : per-pixel world-space origin (xyz; w padded)
//   ray_direction: per-pixel world-space direction (xyz; w padded; need not be unit)
//
// Outputs (storage):
//   out_rgba   : per-pixel composited (R, G, B, alpha=1-T)
//   out_depth  : per-pixel front-most-hit `t` along the ray, or -1
//                if no splat contributed
//
// Uniforms: pixel_count, splat_count, max_distance, min_alpha (the
// per-splat α threshold below which a hit is discarded, matching
// Newton's 1/255 rule).

const NO_HIT_T: f32 = -1.0;
// Top-K cap: per-pixel sorted list of best (front-to-back) hits.
// Larger gives more accurate composites for dense splat clouds at
// the cost of more registers. 16 matches Newton's default
// `gaussians_max_num_hits`.
const MAX_HITS: u32 = 16u;
// Early-out transmittance floor (matches Newton's `gaussians_min_transmittance`).
const MIN_TRANSMITTANCE: f32 = 0.003;
const EPS: f32 = 1e-12;

struct Uniforms {
    pixel_count:  u32,
    splat_count:  u32,
    max_distance: f32,
    min_alpha:    f32,
};

@group(0) @binding(0) var<storage, read>  splat_pos:     array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>  splat_scale:   array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>  splat_rot:     array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>  splat_color:   array<vec4<f32>>;
@group(0) @binding(4) var<storage, read>  splat_opacity: array<f32>;

@group(1) @binding(0) var<storage, read>  ray_origin:    array<vec4<f32>>;
@group(1) @binding(1) var<storage, read>  ray_direction: array<vec4<f32>>;

@group(2) @binding(0) var<storage, read_write> out_rgba:  array<vec4<f32>>;
@group(2) @binding(1) var<storage, read_write> out_depth: array<f32>;

@group(3) @binding(0) var<uniform> uniforms: Uniforms;

// ── Quaternion helper (matches raycast.wgsl) ──────────────────────
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = q.xyz;
    let t = 2.0 * cross(qv, v);
    return v + q.w * t + cross(qv, t);
}

fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}

// ── Splat response in canonical frame. ────────────────────────────
// Returns (alpha, t_hit) where t_hit is the camera-distance parametric
// `t` along the world ray (matches Newton's CAMERA_DISTANCE sort
// mode). A miss returns (0, -1).
fn ray_splat_response(
    splat_idx: u32,
    ro_world: vec3<f32>,
    rd_world: vec3<f32>,
    max_distance: f32,
    min_alpha: f32,
) -> vec2<f32> {
    let centre = splat_pos[splat_idx].xyz;
    let scale  = splat_scale[splat_idx].xyz;
    let rot    = splat_rot[splat_idx];
    let opacity = splat_opacity[splat_idx];

    // Bring ray into canonical (unit-isotropic) frame.
    let inv_rot = quat_conj(rot);
    let centered = ro_world - centre;
    let o_rot = quat_rotate(inv_rot, centered);
    let d_rot = quat_rotate(inv_rot, rd_world);
    let inv_s = vec3<f32>(1.0 / scale.x, 1.0 / scale.y, 1.0 / scale.z);
    let o_local = o_rot * inv_s;
    let d_local = d_rot * inv_s;

    // Closest approach in the canonical frame:
    //   m² = |d × o|² / |d|²
    //   ρ_max = exp(-½ · m²)
    let d2 = dot(d_local, d_local);
    if (d2 < EPS) {
        return vec2<f32>(0.0, NO_HIT_T);
    }
    let crd = cross(d_local, o_local);
    let m2 = dot(crd, crd) / d2;
    let rho = exp(-0.5 * m2);
    let alpha = clamp(opacity * rho, 0.0, 1.0);
    if (alpha < min_alpha) {
        return vec2<f32>(0.0, NO_HIT_T);
    }

    // Hit distance = camera-distance projection of centre on world ray.
    let to_center = centre - ro_world;
    let rd_len2 = dot(rd_world, rd_world);
    if (rd_len2 < EPS) {
        return vec2<f32>(0.0, NO_HIT_T);
    }
    let t = dot(to_center, rd_world) / rd_len2;
    if (t <= 0.0 || t > max_distance) {
        return vec2<f32>(0.0, NO_HIT_T);
    }
    return vec2<f32>(alpha, t);
}

@compute @workgroup_size(64)
fn render(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.pixel_count) { return; }

    let ro = ray_origin[tid].xyz;
    let rd = ray_direction[tid].xyz;

    // ── Top-K mini-sort: front-to-back insertion. ────────────────
    // Arrays of fixed length so the WGSL compiler doesn't need a
    // dynamic stack. MAX_HITS small (16) → O(K) insertion is fine.
    var hit_t:     array<f32, 16>;
    var hit_alpha: array<f32, 16>;
    var hit_color: array<vec3<f32>, 16>;
    for (var k: u32 = 0u; k < MAX_HITS; k = k + 1u) {
        hit_t[k]     = uniforms.max_distance + 1.0;
        hit_alpha[k] = 0.0;
        hit_color[k] = vec3<f32>(0.0);
    }
    var num_hits: u32 = 0u;

    for (var i: u32 = 0u; i < uniforms.splat_count; i = i + 1u) {
        let r = ray_splat_response(i, ro, rd, uniforms.max_distance, uniforms.min_alpha);
        let alpha = r.x;
        let t = r.y;
        if (t < 0.0) { continue; }

        // Insertion sort: find first slot where `t < hit_t[k]`.
        // If we hit MAX_HITS, only insert if smaller than the worst entry.
        var insert_idx: u32 = MAX_HITS;
        for (var k: u32 = 0u; k < MAX_HITS; k = k + 1u) {
            if (t < hit_t[k]) {
                insert_idx = k;
                break;
            }
        }
        if (insert_idx >= MAX_HITS) {
            continue; // not better than current worst
        }

        // Grow the active window (capped at MAX_HITS) and shift right.
        if (num_hits < MAX_HITS) {
            num_hits = num_hits + 1u;
        }
        var k: u32 = num_hits;
        while (k > insert_idx + 1u) {
            let prev = k - 1u;
            hit_t[prev]     = hit_t[prev - 1u];
            hit_alpha[prev] = hit_alpha[prev - 1u];
            hit_color[prev] = hit_color[prev - 1u];
            k = k - 1u;
        }

        let c = splat_color[i].xyz;
        hit_t[insert_idx]     = t;
        hit_alpha[insert_idx] = alpha;
        hit_color[insert_idx] = c;
    }

    // ── α-composite front-to-back. ────────────────────────────────
    var rgb = vec3<f32>(0.0);
    var transmittance: f32 = 1.0;
    var front_t: f32 = NO_HIT_T;
    for (var k: u32 = 0u; k < num_hits; k = k + 1u) {
        let a = hit_alpha[k];
        rgb = rgb + hit_color[k] * a * transmittance;
        if (front_t < 0.0) {
            front_t = hit_t[k];
        }
        transmittance = transmittance * (1.0 - a);
        if (transmittance < MIN_TRANSMITTANCE) {
            break;
        }
    }

    out_rgba[tid] = vec4<f32>(rgb, 1.0 - transmittance);
    out_depth[tid] = front_t;
}
