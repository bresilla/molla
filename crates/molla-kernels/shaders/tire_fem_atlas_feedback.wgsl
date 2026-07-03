// Stage F2(b) — fused-atlas tire contact feedback (bristle friction).
//
// One dispatch runs the per-tire `tire_fem_contact_feedback.wgsl`
// physics core across EVERY tire in a fused atlas batch: one thread
// per atlas node, per-tire parameters from `fb_param_rows`, per-tire
// tread slices from `offset_rows`, and the wheel spin read live from
// the packed hub-state rows (2 rows per tire, same layout as
// `tire_fem_atlas.wgsl`) so the whole 40-wheel substep loop needs no
// per-substep host values.
//
// Flat-ground v1: matches the per-tire resident path with dummy
// terrain maps exactly — ground plane at `p0.z`, normal (0,1,0),
// terrain friction = mu_static (friction_scale ≡ 1), zero water film
// (wet factors ≡ 1 via the same early-return), zero sinkage (no
// rolling-resistance term; the per-tire resident path also runs with
// `soil0.x = 0`). Heightfield support follows the per-tire
// `dispatch_*_with_gpu_terrain` pattern when needed.
//
// Force rows written to `force_out` carry (fx, 0, fz): the hub-param
// bridge (`hub_params_from_feedback_force_rows`) sums full xyz, and
// the hub convention is XZ-only — the normal load reaches the hub
// through the mesh + bead reduction, matching the CPU `step_inner`
// and the per-tire `hub_reduce_with_tread_wrench` behavior.
//
// Velocity updates are IN PLACE (each thread owns its node), so no
// vel_out→vel copy pass is needed, unlike the per-tire resident path.

struct AtlasOffsetRow {
    // x=node_offset, y=node_count, z=node_workgroup_offset, w=node_workgroup_count
    node: vec4<u32>,
    // x=tet_offset, y=tet_count, zw=reserved
    tet: vec4<u32>,
    // x=tread_offset, y=tread_count, z=bead_offset, w=bead_count
    contact: vec4<u32>,
    // x=edge_offset, y=edge_count, z=dihedral_offset, w=dihedral_count
    xpbd: vec4<u32>,
};

struct AtlasFeedbackParamRow {
    // x=dt, y=contact_band, z=ground_y, w=rolling_radius
    p0: vec4<f32>,
    // x=normal_load_per_tread, y=area_per_tread, z=tread_shear_stiffness, w=bristle_damping
    p1: vec4<f32>,
    // x=mu_static, y=mu_dynamic, z=cohesion, w=bristle_relax_rate
    p2: vec4<f32>,
    // xyz=heading, w=omega (used only when uniforms say host-spin)
    p3: vec4<f32>,
};

struct AtlasFeedbackUniforms {
    // x=total node count, y=resident-spin flag (1 → ω from hub_state), zw=reserved
    counts: vec4<u32>,
};

@group(0) @binding(0) var<storage, read>       pos_in:        array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> vel_inout:     array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       mass_in:       array<f32>;
@group(0) @binding(3) var<storage, read>       tread_ids:     array<u32>;
@group(0) @binding(4) var<storage, read>       node_tire_indices: array<u32>;
@group(0) @binding(5) var<storage, read>       offset_rows:   array<AtlasOffsetRow>;
@group(0) @binding(6) var<storage, read>       fb_param_rows: array<AtlasFeedbackParamRow>;
@group(0) @binding(7) var<storage, read>       bristle_in:    array<vec4<f32>>;
@group(0) @binding(8) var<storage, read_write> bristle_out:   array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> force_out:     array<vec4<f32>>;
@group(0) @binding(10) var<storage, read>      hub_state:     array<vec4<f32>>;
@group(1) @binding(0) var<uniform>             u:             AtlasFeedbackUniforms;

// Verbatim from tire_fem_contact_feedback.wgsl — the arithmetic must
// stay expression-for-expression identical for cross-path parity.
fn safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let l2 = dot(v, v);
    if (l2 > 1.0e-14) {
        return v * inverseSqrt(l2);
    }
    return fallback;
}

fn smooth_combined_force(trial_long: f32, trial_lat: f32, static_cap: f32, dynamic_cap: f32) -> vec3<f32> {
    let mag = sqrt(trial_long * trial_long + trial_lat * trial_lat);
    if (mag <= 1.0e-9) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    if (mag <= static_cap) {
        return vec3<f32>(trial_long, trial_lat, 0.0);
    }
    let span = max(static_cap - dynamic_cap, 1.0e-6);
    let t_raw = clamp((mag - static_cap) / span, 0.0, 1.0);
    let t = t_raw * t_raw * (3.0 - 2.0 * t_raw);
    let cap = mix(static_cap, dynamic_cap, t);
    let s = cap / mag;
    return vec3<f32>(trial_long * s, trial_lat * s, 1.0);
}

@compute @workgroup_size(64)
fn atlas_contact_feedback(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.counts.x) {
        return;
    }
    let tire = node_tire_indices[i];
    let row = fb_param_rows[tire];
    let offsets = offset_rows[tire];

    var v_next = vel_inout[i].xyz;
    var force = vec3<f32>(0.0);
    var tread_slot: u32 = 0xffffffffu;
    let tread_begin = offsets.contact.x;
    let tread_end = tread_begin + offsets.contact.y;
    for (var t: u32 = tread_begin; t < tread_end; t = t + 1u) {
        if (tread_ids[t] == i) {
            tread_slot = t;
        }
    }

    if (tread_slot != 0xffffffffu) {
        let p = pos_in[i].xyz;
        // Flat-ground contact sample (per-tire dummy-map path).
        let normal = vec3<f32>(0.0, 1.0, 0.0);
        let clearance = p.y - row.p0.z;
        let band = max(row.p0.y, 1.0e-6);
        let dt = max(row.p0.x, 0.0);
        if (clearance <= band && dt > 0.0) {
            let weight = clamp((band - clearance) / band, 0.0, 3.0);
            let normal_load = max(row.p1.x, 0.0) * weight;
            let heading = safe_normalize(
                row.p3.xyz - normal * dot(row.p3.xyz, normal),
                vec3<f32>(1.0, 0.0, 0.0),
            );
            let side = safe_normalize(cross(heading, normal), vec3<f32>(0.0, 0.0, 1.0));
            let rel_v = v_next; // flat ground: zero surface velocity
            let v_long = dot(rel_v, heading);
            let v_lat = dot(rel_v, side);
            let radius = max(row.p0.w, 1.0e-4);
            var omega_spin = row.p3.w;
            if (u.counts.y > 0u) {
                omega_spin = hub_state[tire * 2u + 1u].z;
            }
            let wheel_speed = omega_spin * radius;
            let slip_speed = wheel_speed - v_long;
            let lateral_slip_speed = -v_lat;

            var b_long = dot(bristle_in[tread_slot].xyz, heading) + slip_speed * dt;
            var b_lat = dot(bristle_in[tread_slot].xyz, side) + lateral_slip_speed * dt;
            let area = max(row.p1.y, 1.0e-8);
            let shear_k = max(row.p1.z, 0.0);
            // Zero water film → wet factors are exactly (1,1,1,1),
            // matching the per-tire shader's early return.
            let damping = max(row.p1.w, 0.0);
            let trial_long = shear_k * area * b_long + damping * area * slip_speed;
            let trial_lat = shear_k * area * b_lat + damping * area * lateral_slip_speed;
            // Terrain friction = mu_static reference → scale ≡ 1.
            let cohesion = max(row.p2.z, 0.0);
            let static_cap = max(row.p2.x, 0.0) * normal_load + cohesion * area;
            let dynamic_cap = max(row.p2.y, 0.0) * normal_load + cohesion * area;
            let ft = smooth_combined_force(trial_long, trial_lat, static_cap, dynamic_cap);
            if (ft.z > 0.5) {
                let max_b = dynamic_cap / max(shear_k * area, 1.0e-9);
                let b_mag = sqrt(b_long * b_long + b_lat * b_lat);
                if (b_mag > max_b && b_mag > 1.0e-12) {
                    let s = max_b / b_mag;
                    b_long = b_long * s;
                    b_lat = b_lat * s;
                }
            }
            let bristle = heading * b_long + side * b_lat;
            bristle_out[tread_slot] = vec4<f32>(bristle, 0.0);
            force = normal * normal_load + heading * ft.x + side * ft.y;
            let inv_m = 1.0 / max(mass_in[i], 1.0e-9);
            v_next = clamp(v_next + force * inv_m * dt, vec3<f32>(-12.0), vec3<f32>(12.0));
        } else {
            let dt_decay = max(row.p0.x, 0.0);
            let decay = exp(-max(row.p2.w, 0.0) * dt_decay);
            bristle_out[tread_slot] = vec4<f32>(bristle_in[tread_slot].xyz * decay, 0.0);
        }
    }

    vel_inout[i] = vec4<f32>(v_next, 0.0);
    // XZ-only hub wrench rows (see header).
    force_out[i] = vec4<f32>(force.x, 0.0, force.z, 0.0);
}
