// Stage 44 v3 — differentiable 3D-Gaussian splat raytrace with
// view-dependent (spherical-harmonics) colour, FORWARD pass.
//
// Identical geometry / α-composite pipeline to splat_raytrace.wgsl;
// the only change is that each splat's RGB is computed from 9 SH
// coefficient triples (bands 0..=2) at the per-pixel view direction
// instead of read from a constant diffuse buffer.
//
// SH coefficient layout (matches GaussianSplatSh3d::sh):
//   splat_sh[i*9 + 0] = c_{0, 0}     (band 0)
//   splat_sh[i*9 + 1] = c_{1,-1}     (band 1)
//   splat_sh[i*9 + 2] = c_{1, 0}
//   splat_sh[i*9 + 3] = c_{1,+1}
//   splat_sh[i*9 + 4] = c_{2,-2}     (band 2)
//   splat_sh[i*9 + 5] = c_{2,-1}
//   splat_sh[i*9 + 6] = c_{2, 0}
//   splat_sh[i*9 + 7] = c_{2,+1}
//   splat_sh[i*9 + 8] = c_{2,+2}
//
// Each coefficient is a vec3 (R, G, B) padded to vec4 (storage
// alignment). The same 9-basis evaluation is used in both the
// forward kernel here and the backward kernel `splat_raytrace_sh_
// backward.wgsl` (and the Rust CPU oracle `eval_sh`).
//
// Inputs (group 0, splats):
//   splat_pos     : per-splat world-space mean (xyz; w padded)
//   splat_scale   : per-splat anisotropic stddev (xyz; w padded)
//   splat_rot     : per-splat unit quaternion (x, y, z, w)
//   splat_sh      : per-splat 9 SH coeffs (vec3 padded to vec4; stride 9)
//   splat_opacity : per-splat bare opacity ∈ [0, 1]
//
// Inputs (group 1, rays):
//   ray_origin    : per-pixel world-space origin (xyz; w padded)
//   ray_direction : per-pixel world-space direction (xyz; w padded)
//
// Outputs (group 2):
//   out_rgba   : per-pixel composited (R, G, B, alpha = 1 − T)
//   out_depth  : per-pixel front-most-hit `t` along the ray, or −1.

const NO_HIT_T: f32 = -1.0;
const MAX_HITS: u32 = 16u;
const MIN_TRANSMITTANCE: f32 = 0.003;
const EPS: f32 = 1e-12;

// SH basis constants (matches Rust SH_C0/.../SH_C2E).
const SH_C0:  f32 =  0.28209479177387814;
const SH_C1:  f32 =  0.4886025119029199;
const SH_C2A: f32 =  1.0925484305920792;
const SH_C2B: f32 = -1.0925484305920792;
const SH_C2C: f32 =  0.31539156525252005;
const SH_C2D: f32 = -1.0925484305920792;
const SH_C2E: f32 =  0.5462742152960396;

const SH_COUNT: u32 = 9u;

struct Uniforms {
    pixel_count:  u32,
    splat_count:  u32,
    max_distance: f32,
    min_alpha:    f32,
};

@group(0) @binding(0) var<storage, read>  splat_pos:     array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>  splat_scale:   array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>  splat_rot:     array<vec4<f32>>;
@group(0) @binding(3) var<storage, read>  splat_sh:      array<vec4<f32>>;
@group(0) @binding(4) var<storage, read>  splat_opacity: array<f32>;

@group(1) @binding(0) var<storage, read>  ray_origin:    array<vec4<f32>>;
@group(1) @binding(1) var<storage, read>  ray_direction: array<vec4<f32>>;

@group(2) @binding(0) var<storage, read_write> out_rgba:  array<vec4<f32>>;
@group(2) @binding(1) var<storage, read_write> out_depth: array<f32>;

@group(3) @binding(0) var<uniform> uniforms: Uniforms;

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = q.xyz;
    let t = 2.0 * cross(qv, v);
    return v + q.w * t + cross(qv, t);
}

fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}

// Evaluate the 9 SH basis functions at the normalised view direction
// `dir`. Returns the basis values as a `array<f32, 9>` (WGSL has no
// dynamic-length local arrays — we use a fixed-length array).
fn sh_basis(dir: vec3<f32>) -> array<f32, 9> {
    var basis: array<f32, 9>;
    let len_sq = dot(dir, dir);
    if (len_sq <= 0.0) {
        basis[0] = SH_C0;
        for (var i: u32 = 1u; i < SH_COUNT; i = i + 1u) {
            basis[i] = 0.0;
        }
        return basis;
    }
    let inv_len = inverseSqrt(len_sq);
    let x = dir.x * inv_len;
    let y = dir.y * inv_len;
    let z = dir.z * inv_len;
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    basis[0] = SH_C0;
    basis[1] = -SH_C1 * y;
    basis[2] =  SH_C1 * z;
    basis[3] = -SH_C1 * x;
    basis[4] =  SH_C2A * (x * y);
    basis[5] =  SH_C2B * (y * z);
    basis[6] =  SH_C2C * (3.0 * zz - 1.0);
    basis[7] =  SH_C2D * (x * z);
    basis[8] =  SH_C2E * (xx - yy);
    return basis;
}

fn eval_sh_color(splat_idx: u32, basis: array<f32, 9>) -> vec3<f32> {
    let base = splat_idx * SH_COUNT;
    var rgb = vec3<f32>(0.0);
    for (var k: u32 = 0u; k < SH_COUNT; k = k + 1u) {
        rgb = rgb + splat_sh[base + k].xyz * basis[k];
    }
    return rgb;
}

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

    let inv_rot = quat_conj(rot);
    let centered = ro_world - centre;
    let o_rot = quat_rotate(inv_rot, centered);
    let d_rot = quat_rotate(inv_rot, rd_world);
    let inv_s = vec3<f32>(1.0 / scale.x, 1.0 / scale.y, 1.0 / scale.z);
    let o_local = o_rot * inv_s;
    let d_local = d_rot * inv_s;

    let d2 = dot(d_local, d_local);
    if (d2 < EPS) { return vec2<f32>(0.0, NO_HIT_T); }
    let crd = cross(d_local, o_local);
    let m2 = dot(crd, crd) / d2;
    let rho = exp(-0.5 * m2);
    let alpha = clamp(opacity * rho, 0.0, 1.0);
    if (alpha < min_alpha) { return vec2<f32>(0.0, NO_HIT_T); }

    let to_center = centre - ro_world;
    let rd_len2 = dot(rd_world, rd_world);
    if (rd_len2 < EPS) { return vec2<f32>(0.0, NO_HIT_T); }
    let t = dot(to_center, rd_world) / rd_len2;
    if (t <= 0.0 || t > max_distance) { return vec2<f32>(0.0, NO_HIT_T); }
    return vec2<f32>(alpha, t);
}

@compute @workgroup_size(64)
fn render(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.pixel_count) { return; }

    let ro = ray_origin[tid].xyz;
    let rd = ray_direction[tid].xyz;
    // SH basis is evaluated once per pixel (view direction = ray dir).
    let basis = sh_basis(rd);

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

        var insert_idx: u32 = MAX_HITS;
        for (var k: u32 = 0u; k < MAX_HITS; k = k + 1u) {
            if (t < hit_t[k]) {
                insert_idx = k;
                break;
            }
        }
        if (insert_idx >= MAX_HITS) {
            continue;
        }

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

        let c = eval_sh_color(i, basis);
        hit_t[insert_idx]     = t;
        hit_alpha[insert_idx] = alpha;
        hit_color[insert_idx] = c;
    }

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
