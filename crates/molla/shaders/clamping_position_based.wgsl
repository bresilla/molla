// Position-based effort clamp via shared lookup table.
//
// Mirrors Newton's `_position_based_clamp_kernel`
// (xtra/newton/_src/actuators/clamping/clamping_position_based.py).
//
// Each actuator looks up its current joint position in a shared
// (pos → max_effort) table and symmetrically clamps the controller
// output to ±lookup. This models actuators whose available effort
// drops near hard joint limits.
//
// Lookup is linear interpolation between adjacent samples; values
// outside the table clamp to the boundary entries.

struct Uniforms {
    actuator_count: u32,
    lookup_size:    u32,
    _pad0:          u32,
    _pad1:          u32,
};

@group(0) @binding(0) var<storage, read>       dof_indices:      array<u32>;
@group(0) @binding(1) var<storage, read>       lookup_positions: array<f32>;
@group(0) @binding(2) var<storage, read>       lookup_efforts:   array<f32>;

@group(1) @binding(0) var<storage, read>       joint_q:     array<f32>;
@group(1) @binding(1) var<storage, read_write> joint_force: array<f32>;

@group(2) @binding(0) var<uniform> uniforms: Uniforms;

fn interp_1d(x: f32) -> f32 {
    let n = uniforms.lookup_size;
    if (n == 0u) { return 0.0; }
    if (x <= lookup_positions[0u]) {
        return lookup_efforts[0u];
    }
    if (x >= lookup_positions[n - 1u]) {
        return lookup_efforts[n - 1u];
    }
    var k: u32 = 0u;
    loop {
        if (k + 1u >= n) { break; }
        let xn = lookup_positions[k + 1u];
        if (xn >= x) {
            let xp = lookup_positions[k];
            let dx = xn - xp;
            if (dx == 0.0) {
                return lookup_efforts[k];
            }
            let t = (x - xp) / dx;
            return lookup_efforts[k] + t * (lookup_efforts[k + 1u] - lookup_efforts[k]);
        }
        k = k + 1u;
    }
    return lookup_efforts[n - 1u];
}

@compute @workgroup_size(64)
fn clamp_position_based(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.actuator_count) { return; }

    let dof = dof_indices[i];
    let q = joint_q[dof];
    let limit = interp_1d(q);
    let f = joint_force[dof];
    joint_force[dof] = clamp(f, -limit, limit);
}
