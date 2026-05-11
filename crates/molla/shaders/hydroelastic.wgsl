// Hydroelastic contact kernel — Stage 22.1.
//
// Soft-on-rigid hydroelastic contact: each rigid surface sample is
// transformed into the soft body's local frame, the soft body's SDF
// is sampled trilinearly, and a per-sample pressure / normal-force
// is written out for downstream consumption (Dahl friction kernel,
// rigid integrator).
//
// Mirror of the CPU `compute_hydroelastic_contact` in
// `geometry/hydroelastic.rs`. Algorithm:
//
//   1. p_world = R_rigid · p_local + t_rigid     (rigid → world)
//   2. p_soft  = R_softᵀ · (p_world − t_soft)    (world → soft)
//   3. sdf     = trilerp(grid, p_soft)
//   4. grad    = central_diff(grid, p_soft)
//   5. if sdf < 0 → pressure = E · |sdf|, force = pressure · area · n
//      else        → output zeros (no contact)
//   6. n_world = R_soft · normalize(grad)
//
// Output buffers:
//   contact_normal[i]       — world-frame outward normal (zero if no contact)
//   contact_point[i]        — world-frame contact point
//   contact_normal_force[i] — pressure · sample area (N) (zero if no contact)
//
// The friction-coefficient and persistent bristle state for the
// downstream Dahl pass are not touched here — the host wires those
// buffers separately.

struct HydroUniforms {
    sample_count: u32,
    nx: u32,
    ny: u32,
    nz: u32,
    cell_size: f32,
    modulus: f32,
    max_pressure: f32,
    culling_distance: f32,
    // SDF grid origin (the (0, 0, 0) sample) in the SOFT body's
    // local frame. The host pre-bakes the analytic-SDF grid into
    // soft-local coords; world→soft is then the soft pose's inverse.
    origin: vec4<f32>,
    // Soft body world translation (subtracted from p_world before
    // the inverse rotation).
    soft_translation: vec4<f32>,
    // Soft body world rotation as a 3×3 (column-major). The kernel
    // applies the INVERSE (transpose for an orthonormal rotation)
    // to map world coords into soft-local.
    rot_col0: vec4<f32>,
    rot_col1: vec4<f32>,
    rot_col2: vec4<f32>,
    rigid_translation: vec4<f32>,
    rigid_rot_col0: vec4<f32>,
    rigid_rot_col1: vec4<f32>,
    rigid_rot_col2: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: HydroUniforms;

@group(0) @binding(1) var<storage, read> sample_local_pos:    array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> sample_local_normal: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> sample_area:         array<f32>;
@group(0) @binding(4) var<storage, read> sdf_data:            array<f32>;

@group(0) @binding(5) var<storage, read_write> contact_normal:       array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> contact_point:        array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> contact_normal_force: array<f32>;

fn soft_rot() -> mat3x3<f32> {
    return mat3x3<f32>(u.rot_col0.xyz, u.rot_col1.xyz, u.rot_col2.xyz);
}
fn rigid_rot() -> mat3x3<f32> {
    return mat3x3<f32>(u.rigid_rot_col0.xyz, u.rigid_rot_col1.xyz, u.rigid_rot_col2.xyz);
}

fn sdf_index(x: u32, y: u32, z: u32) -> u32 {
    return x + y * u.nx + z * u.nx * u.ny;
}

fn sdf_at(x: u32, y: u32, z: u32) -> f32 {
    if (x >= u.nx || y >= u.ny || z >= u.nz) {
        return 1.0e30;
    }
    return sdf_data[sdf_index(x, y, z)];
}

fn trilerp(p_soft: vec3<f32>) -> f32 {
    let inv = 1.0 / u.cell_size;
    let local = (p_soft - u.origin.xyz) * inv;
    if (local.x < 0.0 || local.y < 0.0 || local.z < 0.0) {
        return 1.0e30;
    }
    let xi = i32(floor(local.x));
    let yi = i32(floor(local.y));
    let zi = i32(floor(local.z));
    let nxi = i32(u.nx);
    let nyi = i32(u.ny);
    let nzi = i32(u.nz);
    if (xi >= nxi - 1 || yi >= nyi - 1 || zi >= nzi - 1) {
        return 1.0e30;
    }
    let tx = local.x - f32(xi);
    let ty = local.y - f32(yi);
    let tz = local.z - f32(zi);
    let x = u32(xi);
    let y = u32(yi);
    let z = u32(zi);
    let c000 = sdf_at(x,     y,     z);
    let c100 = sdf_at(x + 1u, y,     z);
    let c010 = sdf_at(x,     y + 1u, z);
    let c110 = sdf_at(x + 1u, y + 1u, z);
    let c001 = sdf_at(x,     y,     z + 1u);
    let c101 = sdf_at(x + 1u, y,     z + 1u);
    let c011 = sdf_at(x,     y + 1u, z + 1u);
    let c111 = sdf_at(x + 1u, y + 1u, z + 1u);
    let c00 = c000 * (1.0 - tx) + c100 * tx;
    let c01 = c001 * (1.0 - tx) + c101 * tx;
    let c10 = c010 * (1.0 - tx) + c110 * tx;
    let c11 = c011 * (1.0 - tx) + c111 * tx;
    let c0 = c00 * (1.0 - ty) + c10 * ty;
    let c1 = c01 * (1.0 - ty) + c11 * ty;
    return c0 * (1.0 - tz) + c1 * tz;
}

fn sdf_grad(p_soft: vec3<f32>) -> vec3<f32> {
    let h = u.cell_size;
    let dx = trilerp(p_soft + vec3<f32>(h, 0.0, 0.0))
           - trilerp(p_soft - vec3<f32>(h, 0.0, 0.0));
    let dy = trilerp(p_soft + vec3<f32>(0.0, h, 0.0))
           - trilerp(p_soft - vec3<f32>(0.0, h, 0.0));
    let dz = trilerp(p_soft + vec3<f32>(0.0, 0.0, h))
           - trilerp(p_soft - vec3<f32>(0.0, 0.0, h));
    return vec3<f32>(dx, dy, dz);
}

@compute @workgroup_size(64)
fn hydro_contact(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.sample_count) { return; }

    // 1. Rigid-local → world.
    let p_local_rigid = sample_local_pos[i].xyz;
    let area = sample_area[i];
    let p_world = rigid_rot() * p_local_rigid + u.rigid_translation.xyz;

    // 2. World → soft-local: subtract soft-body translation, then
    //    apply the inverse rotation (transpose for an orthonormal
    //    rotation). For an axis-aligned soft body (identity rotation)
    //    this collapses to (p_world - soft_translation).
    let r = soft_rot();
    let r_inv = transpose(r);
    let p_soft = r_inv * (p_world - u.soft_translation.xyz);

    let sdf = trilerp(p_soft);
    if (sdf > u.culling_distance || sdf >= 0.0) {
        contact_normal[i] = vec4<f32>(0.0);
        contact_point[i] = vec4<f32>(p_world, 0.0);
        contact_normal_force[i] = 0.0;
        return;
    }
    let pen = -sdf;
    var pressure = u.modulus * pen;
    if (pressure > u.max_pressure) { pressure = u.max_pressure; }

    let g = sdf_grad(p_soft);
    var n: vec3<f32>;
    if (length(g) > 1.0e-12) {
        n = r * normalize(g);
    } else {
        n = rigid_rot() * sample_local_normal[i].xyz;
    }
    let force = pressure * max(area, 0.0);

    contact_normal[i] = vec4<f32>(n, 0.0);
    contact_point[i] = vec4<f32>(p_world, 0.0);
    contact_normal_force[i] = force;
}

// ───────────────────── Stage 41 v2 sub-pipeline 4 ─────────────────────
//
// `reduce_to_anchor` — Stage 39D anchor reduction on GPU.
//
// Mirrors `geometry::hydroelastic::reduce_with_anchor` from the CPU
// path. Reads the per-sample (point, normal, force_magnitude) arrays
// produced by `hydro_contact` (or any equivalent producer) and emits
// a 2-contact "anchor reduced" pair that preserves both net force and
// net moment of the distributed pressure patch.
//
// Algorithm (single workgroup, strided loop over samples in shared
// memory):
//
//   1. Σ (n_i · f_i) → total_force_vec        (force balance)
//   2. Σ p_i / N      → r_centroid             (geometric centroid)
//   3. Σ (f_i · p_i) / Σ f_i → r_cp           (center of pressure)
//   4. Σ (p_i × (n_i · f_i)) → m_total        (moment about origin)
//   5. Solve a 2-D moment-balance system for α; emit
//      `(r_centroid, force_dir, α · |F|)` and `(r_cp, force_dir,
//      (1 - α) · |F|)`.
//
// Output layout (matching `AnchorReducedContact`):
//   anchor_out[0] = centroid contact (vec4: pt.xyz, force_mag)
//   anchor_out[1] = anchor contact   (vec4: pt.xyz, force_mag)
//   anchor_out[2] = force direction  (vec4: n.xyz, total_force_mag)
//
// **Sample-count cap.** Each workgroup processes up to
// `RED_WG_SIZE × (MAX_STRIDED_ITERS)` samples in a strided loop. The
// shader caps at the uniform's `sample_count` so it's safe for any
// length; the cap is just a performance hint (one workgroup, so
// O(sample_count / 256) loop iterations, fine up to ~64K samples).

const RED_WG_SIZE: u32 = 256u;

// Uniforms for the anchor-reduction pass. Decoupled from
// `HydroUniforms` because the sample count is the only field shared.
struct AnchorUniforms {
    sample_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(1) @binding(0) var<uniform> anchor_u: AnchorUniforms;

// Inputs are the SAME physical buffers as `contact_normal/point/normal_force`
// from the pressure pass — so we declare them as `read_write` here too
// (the access stays read-only at the WGSL level; the usage flag just
// has to be the *exclusive* STORAGE_READ_WRITE so the validator
// doesn't see a read-only + read-write conflict across bind groups
// when both groups bind the same physical buffer).
@group(1) @binding(1) var<storage, read_write> red_point:     array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> red_normal:    array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> red_force_mag: array<f32>;

// Output: 3 vec4 entries (centroid, anchor, force-direction summary).
@group(1) @binding(4) var<storage, read_write> anchor_out: array<vec4<f32>>;

// Workgroup-shared accumulators. Each lane keeps a partial; a tree
// reduction collapses to lane 0. Padded to vec4 for 16-byte
// alignment of the storage class.
var<workgroup> sw_total_force:  array<vec3<f32>, RED_WG_SIZE>;
var<workgroup> sw_centroid_pos: array<vec3<f32>, RED_WG_SIZE>;
var<workgroup> sw_weighted_pos: array<vec3<f32>, RED_WG_SIZE>;
var<workgroup> sw_force_sum:    array<f32,       RED_WG_SIZE>;
var<workgroup> sw_moment:       array<vec3<f32>, RED_WG_SIZE>;
var<workgroup> sw_count:        array<u32,       RED_WG_SIZE>;

@compute @workgroup_size(RED_WG_SIZE)
fn reduce_to_anchor(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let lane = lid.x;
    let n = anchor_u.sample_count;

    // ─── Strided accumulation phase ─────────────────────────────────
    var tf:  vec3<f32> = vec3<f32>(0.0);   // Σ n_i · f_i
    var cp:  vec3<f32> = vec3<f32>(0.0);   // Σ p_i  (geom centroid sum)
    var wp:  vec3<f32> = vec3<f32>(0.0);   // Σ p_i · f_i (center of pressure sum)
    var fs:  f32       = 0.0;              // Σ f_i
    var mm:  vec3<f32> = vec3<f32>(0.0);   // Σ p_i × (n_i · f_i)
    var cn:  u32       = 0u;               // count of non-zero-force samples

    var i: u32 = lane;
    loop {
        if (i >= n) { break; }
        let f_mag = red_force_mag[i];
        // Skip zero-force entries (consistent with CPU path that only
        // populates non-zero contacts in the patch — Stage 22.1 GPU
        // pass zeroes out non-penetrating samples).
        if (f_mag > 0.0) {
            let p = red_point[i].xyz;
            let nrm = red_normal[i].xyz;
            let f_vec = nrm * f_mag;
            tf = tf + f_vec;
            cp = cp + p;
            wp = wp + p * f_mag;
            fs = fs + f_mag;
            mm = mm + cross(p, f_vec);
            cn = cn + 1u;
        }
        i = i + RED_WG_SIZE;
    }
    sw_total_force[lane]  = tf;
    sw_centroid_pos[lane] = cp;
    sw_weighted_pos[lane] = wp;
    sw_force_sum[lane]    = fs;
    sw_moment[lane]       = mm;
    sw_count[lane]        = cn;
    workgroupBarrier();

    // ─── Tree reduction over the workgroup ──────────────────────────
    var stride: u32 = RED_WG_SIZE / 2u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) {
            sw_total_force[lane]  = sw_total_force[lane]  + sw_total_force[lane + stride];
            sw_centroid_pos[lane] = sw_centroid_pos[lane] + sw_centroid_pos[lane + stride];
            sw_weighted_pos[lane] = sw_weighted_pos[lane] + sw_weighted_pos[lane + stride];
            sw_force_sum[lane]    = sw_force_sum[lane]    + sw_force_sum[lane + stride];
            sw_moment[lane]       = sw_moment[lane]       + sw_moment[lane + stride];
            sw_count[lane]        = sw_count[lane]        + sw_count[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // ─── Lane 0: closed-form α solve + emit ─────────────────────────
    if (lane == 0u) {
        let total_force_vec = sw_total_force[0];
        let total_force_mag = length(total_force_vec);
        let count = sw_count[0];
        let cp_sum = sw_centroid_pos[0];
        let wp_sum = sw_weighted_pos[0];
        let f_sum  = sw_force_sum[0];
        let m_total = sw_moment[0];

        if (count == 0u || total_force_mag < 1.0e-20) {
            // Empty manifold or zero net force — emit zeros.
            anchor_out[0] = vec4<f32>(0.0);
            anchor_out[1] = vec4<f32>(0.0);
            anchor_out[2] = vec4<f32>(0.0);
            return;
        }
        let force_dir = total_force_vec / total_force_mag;
        let r_centroid = cp_sum / f32(count);
        var r_cp: vec3<f32>;
        if (f_sum > 1.0e-20) {
            r_cp = wp_sum / f_sum;
        } else {
            r_cp = r_centroid;
        }

        // Solve for α (the centroid's fraction of the total force).
        let dr = r_centroid - r_cp;
        let dr_cross_n = cross(dr, force_dir);
        let denom = dot(dr_cross_n, dr_cross_n) * total_force_mag;
        var alpha: f32;
        if (denom > 1.0e-20) {
            let target_m = m_total - cross(r_cp, force_dir) * total_force_mag;
            alpha = dot(target_m, dr_cross_n) / denom;
        } else {
            alpha = 1.0;
        }
        let f_c_mag = alpha * total_force_mag;
        let f_a_mag = (1.0 - alpha) * total_force_mag;

        // Pack: vec4 = (point.x, point.y, point.z, force_magnitude).
        anchor_out[0] = vec4<f32>(r_centroid, f_c_mag);
        anchor_out[1] = vec4<f32>(r_cp,       f_a_mag);
        // Summary row: force direction + |F_total|.
        anchor_out[2] = vec4<f32>(force_dir,  total_force_mag);
    }
}
