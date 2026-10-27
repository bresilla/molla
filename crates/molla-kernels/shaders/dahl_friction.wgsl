// Dahl friction kernel — Stage 20.1.
//
// Smooth stick→slip friction for particle-vs-rigid contacts. Mirrors
// Newton's `_dahl_friction_kernel` but expressed at the particle
// level rather than at the joint level (Newton's per-joint Dahl is a
// curvature-space thing for cable bending; ours is a tangent-velocity
// thing for cable-on-rigid contacts). The two share the same first-
// order ODE structure:
//
//     dz/dt = v_t · (1 - sgn(v_t) · z / z_max)
//
// where
//   v_t   = relative tangential velocity at the contact (m/s)
//   z     = the persistent bristle-deflection state (m)
//   z_max = μ · F_n / sigma0 (the stick-displacement saturation)
//
// The corresponding friction force is `F_t = -sigma0 · z`, bounded
// by `|F_t| <= μ · F_n` (the Coulomb cone). For the molla v1 use
// case (CABLE on a rigid surface) we only model the 2D in-plane
// tangent, treating each axis independently — so the per-particle
// state is a `vec2`, not a `vec3`.
//
// References:
//   Dahl (1968), "A Solid Friction Model".
//   Newton's `compute_cable_dahl_parameters` and
//   `update_cable_dahl_state` in `xtra/newton/_src/solvers/vbd/
//   rigid_vbd_kernels.py`. Newton uses an exp-decay closed-form
//   integration; we use the simpler explicit Euler step — adequate
//   at the substep dt molla uses (~1 ms).
//
// The kernel is generic in shape: it takes opaque contact data
// (position, normal, tangent velocity, normal force, friction
// coefficient) so Stage 22.1's hydroelastic path can reuse it for
// distributed-pressure friction without changing the interface.

struct DahlUniforms {
    /// Number of contact slots in the input arrays.
    contact_count: u32,
    /// Time step (s).
    dt: f32,
    /// Tangent stiffness sigma0 (N/m). Larger = sharper stick→slip
    /// transition. ~1e4–1e6 N/m is typical for "rubber-on-steel".
    sigma0: f32,
    /// Visco-elastic damping sigma1 (N·s/m). Set 0.0 for the pure
    /// Dahl model; >0 introduces rate-dependent damping.
    sigma1: f32,
    /// Coulomb-cone safety pad: when `μ · F_n < epsilon` the kernel
    /// resets the bristle state to 0 (no friction can be sustained).
    /// 1e-9 N is a sensible default.
    f_normal_eps: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> u: DahlUniforms;

// Inputs (read-only).
/// Per-contact unit normal in world frame.
@group(0) @binding(1) var<storage, read> contact_normal: array<vec4<f32>>;
/// Per-contact relative tangential velocity (already projected onto
/// the contact's tangent plane by the host).
@group(0) @binding(2) var<storage, read> contact_v_tan:  array<vec4<f32>>;
/// Per-contact normal force magnitude (positive when the contact is
/// active / particle is being pushed out of the surface).
@group(0) @binding(3) var<storage, read> contact_f_n:    array<f32>;
/// Per-contact friction coefficient μ (Coulomb).
@group(0) @binding(4) var<storage, read> contact_mu:     array<f32>;

// State (read-write, persistent across substeps for hysteresis).
/// Per-contact bristle deflection state `z` (m), expressed in the
/// world frame so it survives a particle reorientation gracefully.
@group(0) @binding(5) var<storage, read_write> contact_z: array<vec4<f32>>;

// Output (write-only).
/// Per-contact friction force in world frame (N). Add this to the
/// particle force accumulator on the host side; it already includes
/// the Coulomb-cone clamp.
@group(0) @binding(6) var<storage, read_write> contact_friction_force: array<vec4<f32>>;

@compute @workgroup_size(64)
fn dahl_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.contact_count) { return; }

    let f_n = contact_f_n[i];
    let mu = contact_mu[i];
    let f_max = mu * f_n;

    // No active contact / no friction budget → reset bristle state.
    if (f_max < u.f_normal_eps) {
        contact_z[i] = vec4<f32>(0.0);
        contact_friction_force[i] = vec4<f32>(0.0);
        return;
    }

    let n = contact_normal[i].xyz;
    var v_t = contact_v_tan[i].xyz;
    // Project velocity onto the tangent plane (host typically does
    // this already; do it again for safety against accumulating
    // numerical normal drift).
    v_t = v_t - n * dot(v_t, n);

    var z = contact_z[i].xyz;
    // Project z onto the tangent plane too — if the contact normal
    // rotated, z is stale.
    z = z - n * dot(z, n);

    // Per-axis Dahl ODE — explicit Euler. The geometric trick: the
    // term `(1 - sgn(v_t) · z / z_max)` smoothly transitions from
    // 1 (deep in stick — z << z_max) to 0 (slipping — z = z_max),
    // giving the textbook hysteresis loop without an `if (slip)`
    // branch.
    let z_max_scalar = f_max / u.sigma0;

    var dz: vec3<f32>;
    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        let vti = v_t[axis];
        let zi = z[axis];
        let s = select(-1.0, 1.0, vti >= 0.0);
        let saturation = max(0.0, 1.0 - s * zi / max(z_max_scalar, 1e-12));
        dz[axis] = vti * saturation;
    }
    var z_new = z + dz * u.dt;

    // Clamp to the Coulomb cone: |z_new| <= z_max.
    let z_len = length(z_new);
    if (z_len > z_max_scalar) {
        z_new = z_new * (z_max_scalar / max(z_len, 1e-12));
    }

    let f_t = -(u.sigma0 * z_new + u.sigma1 * dz);

    // Hard Coulomb-cone safety clamp on the output (sigma1 can push
    // the magnitude past the cone if the user picks aggressive
    // visco-elastic gain).
    let f_t_len = length(f_t);
    var f_t_out = f_t;
    if (f_t_len > f_max) {
        f_t_out = f_t * (f_max / max(f_t_len, 1e-12));
    }

    contact_z[i] = vec4<f32>(z_new, 0.0);
    contact_friction_force[i] = vec4<f32>(f_t_out, 0.0);
}
