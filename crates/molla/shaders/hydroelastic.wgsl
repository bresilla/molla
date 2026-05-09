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
