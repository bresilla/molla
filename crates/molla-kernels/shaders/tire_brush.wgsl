// Stage (i) — GPU brush-tire contact kernel.
//
// Mirrors `molla_vehicle::real_tire::evaluate_brush_tire`. One
// workgroup per wheel; threads cover circumferential segments and width strips.
// Each segment thread computes its own (penetration, normal load,
// brush traction, sliding flag) and the workgroup reduces them into
// a single per-wheel output row.
//
// Two terrain modes via the `cfg.x` flag in `WheelInput.extra`:
//
//   * `terrain_mode == 0.0` — flat floor at `y = 0`. No heightfield
//     buffer required; the simple/default path.
//   * `terrain_mode == 1.0` — sample a regular heightfield grid stored
//     in the `terrain_heights` storage buffer. The grid is laid out as
//     `terrain_heights[iz * nx + ix]` covering world XZ from
//     `terrain.origin` with spacing `terrain.spacing`. Optional
//     `terrain_friction` samples are bilinearly interpolated at each
//     contact cell and scale the per-wheel material friction, matching
//     the CPU `Terrain::friction_at / friction` path.

const SEGMENTS_MAX: u32 = 256u;
const WG: u32 = 64u;

struct WheelInput {
    // (hub_x, hub_y, hub_z, width_segments; zero selects three strips)
    hub_pos: vec4<f32>,
    // (vx, vy, vz, _)
    hub_vel: vec4<f32>,
    // (hx, _, hz, _) — unit ground-projected heading
    heading: vec4<f32>,
    // x=omega (rad/s), y=radius (m), z=width (m), w=pressure_pa
    spin: vec4<f32>,
    // x=segments (u32 cast to f32), y=carcass_normal_stiffness,
    // z=tread_shear_stiffness, w=relaxation_speed
    tire: vec4<f32>,
    // x=mu_static, y=mu_dynamic, z=bristle_damping, w=rolling_resistance
    material: vec4<f32>,
    // x=cohesion, y=terrain_mode (0=flat, 1=heightfield), z=mu_factor
    // (combined wear + thermal correction), w=camber_angle_rad
    extra: vec4<f32>,
};

struct TerrainUniforms {
    // x=origin_x, y=origin_z, z=spacing, w=reference friction
    origin: vec4<f32>,
    // x=nx, y=nz, z=enabled (0/1), w=friction map enabled (0/1)
    dims: vec4<u32>,
};

struct WheelOutput {
    // x=Fx (longitudinal in heading frame), y=normal_load,
    // z=Fy (lateral in side frame),
    // w=longitudinal_force (Fx - rolling resistance)
    forces: vec4<f32>,
    // x=patch_length, y=slip_ratio, z=slip_speed, w=contact_count
    diag1: vec4<f32>,
    // x=sliding_count, y=aligning_moment_z, z=lateral_slip_speed,
    // w=slip_angle_rad
    diag2: vec4<f32>,
    // x=dissipated_power_w, y=camber_angle_rad,
    // z=contact_center_x, w=pneumatic_trail_m
    diag3: vec4<f32>,
    // x=adhesion_fraction, y=sliding_fraction,
    // z=available_friction_budget, w=consumed_friction_budget
    diag4: vec4<f32>,
    // x=effective_longitudinal_stiffness, y=effective_cornering_stiffness
    diag5: vec4<f32>,
    // xyz=world force accumulated from per-contact terrain normals
    world_force: vec4<f32>,
};

@group(0) @binding(0) var<storage, read>       wheels:           array<WheelInput>;
@group(0) @binding(1) var<storage, read_write> outputs:          array<WheelOutput>;
@group(0) @binding(2) var<storage, read>       terrain_heights:  array<f32>;
@group(0) @binding(3) var<storage, read>       terrain_friction: array<f32>;
@group(1) @binding(0) var<uniform>             terrain_u:        TerrainUniforms;

fn sample_terrain(world_x: f32, world_z: f32) -> f32 {
    if (terrain_u.dims.z == 0u) {
        return 0.0;
    }
    let nx = terrain_u.dims.x;
    let nz = terrain_u.dims.y;
    let spacing = max(terrain_u.origin.z, 1.0e-6);
    let fx = (world_x - terrain_u.origin.x) / spacing;
    let fz = (world_z - terrain_u.origin.y) / spacing;
    let ix0_i = i32(floor(fx));
    let iz0_i = i32(floor(fz));
    let tx = fx - f32(ix0_i);
    let tz = fz - f32(iz0_i);
    let ix0 = u32(clamp(ix0_i, 0, i32(nx) - 1));
    let iz0 = u32(clamp(iz0_i, 0, i32(nz) - 1));
    let ix1 = u32(clamp(ix0_i + 1, 0, i32(nx) - 1));
    let iz1 = u32(clamp(iz0_i + 1, 0, i32(nz) - 1));
    let h00 = terrain_heights[iz0 * nx + ix0];
    let h10 = terrain_heights[iz0 * nx + ix1];
    let h01 = terrain_heights[iz1 * nx + ix0];
    let h11 = terrain_heights[iz1 * nx + ix1];
    let h0 = h00 * (1.0 - tx) + h10 * tx;
    let h1 = h01 * (1.0 - tx) + h11 * tx;
    return h0 * (1.0 - tz) + h1 * tz;
}

fn safe_normalize(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let l = length(v);
    if (l > 1.0e-8) {
        return v / l;
    }
    return fallback;
}

fn sample_terrain_normal(world_x: f32, world_z: f32) -> vec3<f32> {
    if (terrain_u.dims.z == 0u) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    let d = max(terrain_u.origin.z * 0.5, 1.0e-5);
    let dhdx = (sample_terrain(world_x + d, world_z) - sample_terrain(world_x - d, world_z)) / (2.0 * d);
    let dhdz = (sample_terrain(world_x, world_z + d) - sample_terrain(world_x, world_z - d)) / (2.0 * d);
    return safe_normalize(vec3<f32>(-dhdx, 1.0, -dhdz), vec3<f32>(0.0, 1.0, 0.0));
}

fn sample_terrain_friction(world_x: f32, world_z: f32) -> f32 {
    if (terrain_u.dims.z == 0u || terrain_u.dims.w == 0u) {
        return max(terrain_u.origin.w, 1.0e-6);
    }
    let nx = terrain_u.dims.x;
    let nz = terrain_u.dims.y;
    let spacing = max(terrain_u.origin.z, 1.0e-6);
    let fx = (world_x - terrain_u.origin.x) / spacing;
    let fz = (world_z - terrain_u.origin.y) / spacing;
    let ix0_i = i32(floor(fx));
    let iz0_i = i32(floor(fz));
    let tx = fx - f32(ix0_i);
    let tz = fz - f32(iz0_i);
    let ix0 = u32(clamp(ix0_i, 0, i32(nx) - 1));
    let iz0 = u32(clamp(iz0_i, 0, i32(nz) - 1));
    let ix1 = u32(clamp(ix0_i + 1, 0, i32(nx) - 1));
    let iz1 = u32(clamp(iz0_i + 1, 0, i32(nz) - 1));
    let f00 = max(terrain_friction[iz0 * nx + ix0], 0.0);
    let f10 = max(terrain_friction[iz0 * nx + ix1], 0.0);
    let f01 = max(terrain_friction[iz1 * nx + ix0], 0.0);
    let f11 = max(terrain_friction[iz1 * nx + ix1], 0.0);
    let f0 = f00 * (1.0 - tx) + f10 * tx;
    let f1 = f01 * (1.0 - tx) + f11 * tx;
    return max(f0 * (1.0 - tz) + f1 * tz, 0.0);
}

var<workgroup> seg_fx:    array<f32, WG>;
var<workgroup> seg_fy:    array<f32, WG>;
var<workgroup> seg_mz:    array<f32, WG>;
var<workgroup> seg_fn:    array<f32, WG>;
var<workgroup> seg_min_x: array<f32, WG>;
var<workgroup> seg_max_x: array<f32, WG>;
var<workgroup> seg_count: array<u32, WG>;
var<workgroup> seg_slide: array<u32, WG>;
var<workgroup> seg_power: array<f32, WG>;
var<workgroup> seg_fn_x: array<f32, WG>;
var<workgroup> seg_lat_x: array<f32, WG>;
var<workgroup> seg_available: array<f32, WG>;
var<workgroup> seg_consumed: array<f32, WG>;
var<workgroup> seg_world_x: array<f32, WG>;
var<workgroup> seg_world_y: array<f32, WG>;
var<workgroup> seg_world_z: array<f32, WG>;
var<workgroup> seg_rolling: array<f32, WG>;
// Patch extent shared across the workgroup for the linear leading-edge
// bristle profile (reduced from the geometry pre-pass before forces).
var<workgroup> wg_patch_min_x: f32;
var<workgroup> wg_patch_max_x: f32;

fn smoothstep01(t_in: f32) -> f32 {
    let t = clamp(t_in, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn brush_load_mu_factor(normal_pressure: f32) -> f32 {
    let load_ratio = clamp(max(normal_pressure, 1.0) / 220000.0, 0.15, 6.0);
    return clamp(pow(load_ratio, -0.07), 0.72, 1.12);
}

fn brush_load_stiffness_factor(normal_pressure: f32) -> f32 {
    let load_ratio = clamp(max(normal_pressure, 1.0) / 220000.0, 0.15, 6.0);
    return clamp(pow(load_ratio, 0.25), 0.55, 1.45);
}

fn brush_pressure_stiffness_factor(pressure: f32) -> f32 {
    let pressure_ratio = clamp(max(pressure, 1.0) / 180000.0, 0.25, 2.50);
    return clamp(pow(pressure_ratio, 0.14), 0.78, 1.14);
}

@compute @workgroup_size(WG)
fn brush_step(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let wi = wid.x;
    let i = lid.x;
    let in_ = wheels[wi];
    let segments = u32(clamp(in_.tire.x, 8.0, f32(SEGMENTS_MAX)));
    let width_segments = u32(clamp(select(3.0, in_.hub_pos.w, in_.hub_pos.w > 0.0), 1.0, 16.0));
    let cells = segments * width_segments;
    let radius = in_.spin.y;
    let width = in_.spin.z;
    let pressure = max(in_.spin.w, 0.0);
    let omega = in_.spin.x;
    let carcass_k = in_.tire.y;
    let shear_k = in_.tire.z;
    let relax = max(in_.tire.w, 1.0e-4);
    // Stage (j) — wear/heat correction multiplies the friction caps.
    let mu_factor = max(in_.extra.z, 1.0e-3);
    let mu_s = in_.material.x * mu_factor;
    let mu_d = in_.material.y * mu_factor;
    let bristle_damp = in_.material.z;
    let rolling = in_.material.w;
    let cohesion = in_.extra.x;
    let terrain_mode = in_.extra.y;
    let camber_angle = clamp(in_.extra.w, -0.7, 0.7);

    let structural_lateral_bias = clamp(in_.heading.w, -0.08, 0.08);
    let heading = normalize(vec3<f32>(in_.heading.x, 0.0, in_.heading.z));
    let side = normalize(cross(heading, vec3<f32>(0.0, 1.0, 0.0)));
    let v_long = dot(in_.hub_vel.xyz, heading);
    let v_lat = dot(in_.hub_vel.xyz, side);
    let wheel_speed = omega * radius;
    let slip_speed = wheel_speed - v_long;
    let denom = max(abs(v_long), relax);
    let slip_ratio = slip_speed / denom;
    let lateral_slip_speed = -v_lat;
    let slip_angle = atan2(v_lat, max(abs(v_long), relax));
    let dtheta = 6.2831853 / f32(segments);
    let arc_len = radius * dtheta;
    let cell_area = width / f32(width_segments) * arc_len;
    let normal_per_area_per_m = carcass_k + 3.0 * pressure / max(radius, 1.0e-6);

    // Initialise scratch for this segment slot.
    if (i < WG) {
        seg_fx[i] = 0.0;
        seg_fy[i] = 0.0;
        seg_mz[i] = 0.0;
        seg_fn[i] = 0.0;
        seg_min_x[i] = 1.0e9;
        seg_max_x[i] = -1.0e9;
        seg_count[i] = 0u;
        seg_slide[i] = 0u;
        seg_power[i] = 0.0;
        seg_fn_x[i] = 0.0;
        seg_lat_x[i] = 0.0;
        seg_available[i] = 0.0;
        seg_consumed[i] = 0.0;
        seg_world_x[i] = 0.0;
        seg_world_y[i] = 0.0;
        seg_world_z[i] = 0.0;
        seg_rolling[i] = 0.0;
    }
    workgroupBarrier();

    // Geometry pre-pass: the linear leading-edge bristle profile needs the
    // patch extent and travel direction before forces are evaluated.
    // Mirrors the CPU pre-pass in `evaluate_brush_tire_oriented`.
    var pre = i;
    loop {
        if (pre >= cells) { break; }
        let theta_p = f32(pre / width_segments) * dtheta;
        let local_x_p = radius * sin(theta_p);
        let local_y_p = -radius * cos(theta_p);
        let local_z_p = ((f32(pre % width_segments) + 0.5) / f32(width_segments) - 0.5) * width;
        let world_x_p = in_.hub_pos.x + heading.x * local_x_p + side.x * local_z_p;
        let world_z_p = in_.hub_pos.z + heading.z * local_x_p + side.z * local_z_p;
        var ground_y_p: f32 = 0.0;
        if (terrain_mode > 0.5) {
            ground_y_p = sample_terrain(world_x_p, world_z_p);
        }
        if (ground_y_p - (in_.hub_pos.y + local_y_p) > 0.0) {
            seg_min_x[i] = min(seg_min_x[i], local_x_p);
            seg_max_x[i] = max(seg_max_x[i], local_x_p);
        }
        pre = pre + WG;
    }
    workgroupBarrier();
    if (i == 0u) {
        var pmin: f32 = 1.0e9;
        var pmax: f32 = -1.0e9;
        var pk: u32 = 0u;
        loop {
            if (pk >= WG) { break; }
            pmin = min(pmin, seg_min_x[pk]);
            pmax = max(pmax, seg_max_x[pk]);
            pk = pk + 1u;
        }
        wg_patch_min_x = pmin;
        wg_patch_max_x = pmax;
    }
    workgroupBarrier();
    let patch_extent = max(wg_patch_max_x - wg_patch_min_x, 0.0);
    let leading_x = select(wg_patch_min_x, wg_patch_max_x, v_long + wheel_speed >= 0.0);
    // Patch-length gain vs the 180 kPa reference inflation: a ∝ k_n^(-1/3).
    let k_ref = max(carcass_k + 3.0 * 180000.0 / max(radius, 1.0e-6), 1.0);
    let patch_gain = clamp(
        pow(k_ref / max(normal_per_area_per_m, 1.0), 1.0 / 3.0),
        0.55,
        1.80,
    );

    // Each thread iterates over multiple segments (strided) so we
    // support segments > WG with a single workgroup.
    var idx = i;
    loop {
        if (idx >= cells) { break; }
        let theta = f32(idx / width_segments) * dtheta;
        let local_x = radius * sin(theta);
        let local_y = -radius * cos(theta);
        let local_z = ((f32(idx % width_segments) + 0.5) / f32(width_segments) - 0.5) * width;
        let world_x = in_.hub_pos.x + heading.x * local_x + side.x * local_z;
        let world_z = in_.hub_pos.z + heading.z * local_x + side.z * local_z;
        let point_y = in_.hub_pos.y + local_y;
        // Terrain mode 0 = flat floor at y = 0; mode 1 = bilinear
        // heightfield lookup.
        var ground_y: f32 = 0.0;
        if (terrain_mode > 0.5) {
            ground_y = sample_terrain(world_x, world_z);
        }
        let pen = ground_y - point_y;
        if (pen > 0.0) {
            let cell_normal = sample_terrain_normal(world_x, world_z);
            let cell_heading = safe_normalize(
                heading - cell_normal * dot(heading, cell_normal),
                heading,
            );
            let cell_side = safe_normalize(cross(cell_heading, cell_normal), side);
            let v_long_cell = dot(in_.hub_vel.xyz, cell_heading);
            let v_lat_cell = dot(in_.hub_vel.xyz, cell_side);
            let denom_cell = max(abs(v_long_cell), relax);
            let slip_speed_cell = wheel_speed - v_long_cell;
            let slip_ratio_cell = slip_speed_cell / denom_cell;
            let lateral_slip_speed_cell = -v_lat_cell;
            var friction_scale: f32 = 1.0;
            if (terrain_mode > 0.5 && terrain_u.dims.w != 0u) {
                friction_scale = clamp(
                    sample_terrain_friction(world_x, world_z) / max(terrain_u.origin.w, 1.0e-6),
                    0.0,
                    3.0,
                );
            }
            let camber_bias = clamp(
                1.0 + 0.35 * camber_angle * clamp(local_x / max(radius, 1.0e-6), -1.0, 1.0),
                0.35,
                1.65,
            );
            let fn_i = normal_per_area_per_m * pen * cell_area * camber_bias;
            let normal_pressure = fn_i / max(cell_area, 1.0e-9);
            let load_mu = brush_load_mu_factor(normal_pressure);
            let shear_k_eff = shear_k
                * brush_load_stiffness_factor(normal_pressure)
                * brush_pressure_stiffness_factor(pressure);
            let cell_len = max(arc_len, 1.0e-6);
            let profile = patch_gain
                * clamp(
                    2.0 * (abs(leading_x - local_x) + 0.5 * cell_len)
                        / (patch_extent + cell_len),
                    0.0,
                    4.0,
                );
            let bristle_disp_long = slip_ratio_cell * radius * profile;
            let bristle_disp_lat = lateral_slip_speed_cell / denom_cell * radius * profile;
            let camber_thrust = -0.22 * camber_angle * shear_k_eff * cell_area;
            let structural_bias = structural_lateral_bias * fn_i;
            let trial_long = shear_k_eff * cell_area * bristle_disp_long
                + bristle_damp * cell_area * slip_speed_cell;
            let trial_lat = shear_k_eff * cell_area * bristle_disp_lat
                + bristle_damp * cell_area * lateral_slip_speed_cell
                + camber_thrust
                + structural_bias;
            let static_cap = max(mu_s * friction_scale * load_mu * fn_i + cohesion * cell_area, 0.0);
            let dynamic_cap = min(max(mu_d * friction_scale * load_mu * fn_i + cohesion * cell_area, 0.0), static_cap);
            let trial_mag = sqrt(trial_long * trial_long + trial_lat * trial_lat);
            var ft_long: f32;
            var ft_lat: f32;
            var sliding: u32;
            var consumed_i: f32;
            if (trial_mag > 1.0e-12 && static_cap > 1.0e-12) {
                let util = trial_mag / max(static_cap, 1.0e-12);
                let slide_blend = smoothstep01((util - 0.75) / 0.75);
                let cap = max(static_cap + (dynamic_cap - static_cap) * slide_blend, 1.0e-12);
                let u = trial_mag / cap;
                let scale = pow(1.0 + pow(u, 8.0), -0.125);
                ft_long = trial_long * scale;
                ft_lat = trial_lat * scale;
                consumed_i = trial_mag * scale;
                sliding = select(0u, 1u, util > 0.90 || consumed_i < trial_mag * 0.995);
            } else {
                ft_long = 0.0;
                ft_lat = 0.0;
                consumed_i = 0.0;
                sliding = 0u;
            }
            seg_fx[i] = seg_fx[i] + ft_long;
            seg_fy[i] = seg_fy[i] + ft_lat;
            let rolling_i = rolling * fn_i * sign(v_long_cell);
            seg_rolling[i] = seg_rolling[i] + rolling_i;
            let world_force = cell_heading * (ft_long - rolling_i) + cell_side * ft_lat + cell_normal * fn_i;
            seg_world_x[i] = seg_world_x[i] + world_force.x;
            seg_world_y[i] = seg_world_y[i] + world_force.y;
            seg_world_z[i] = seg_world_z[i] + world_force.z;
            seg_available[i] = seg_available[i] + static_cap;
            seg_consumed[i] = seg_consumed[i] + consumed_i;
            seg_mz[i] = seg_mz[i] + local_z * (ft_long - rolling_i) - local_x * ft_lat;
            seg_fn[i] = seg_fn[i] + fn_i;
            seg_min_x[i] = min(seg_min_x[i], local_x);
            seg_max_x[i] = max(seg_max_x[i], local_x);
            seg_fn_x[i] = seg_fn_x[i] + local_x * fn_i;
            seg_lat_x[i] = seg_lat_x[i] + local_x * abs(ft_lat);
            seg_count[i] = seg_count[i] + 1u;
            seg_slide[i] = seg_slide[i] + sliding;
            if (sliding > 0u) {
                let ft = sqrt(ft_long * ft_long + ft_lat * ft_lat);
                let vs = sqrt(slip_speed_cell * slip_speed_cell + lateral_slip_speed_cell * lateral_slip_speed_cell);
                seg_power[i] = seg_power[i] + ft * vs;
            }
        }
        idx = idx + WG;
    }
    workgroupBarrier();

    // Workgroup reduction (linear, single thread).
    if (i == 0u) {
        var total_fx: f32 = 0.0;
        var total_fy: f32 = 0.0;
        var total_mz: f32 = 0.0;
        var total_fn: f32 = 0.0;
        var min_x: f32 = 1.0e9;
        var max_x: f32 = -1.0e9;
        var total_count: u32 = 0u;
        var total_slide: u32 = 0u;
        var total_power: f32 = 0.0;
        var total_fn_x: f32 = 0.0;
        var total_lat_x: f32 = 0.0;
        var total_available: f32 = 0.0;
        var total_consumed: f32 = 0.0;
        var total_world: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
        var rolling_force: f32 = 0.0;
        var k: u32 = 0u;
        loop {
            if (k >= WG) { break; }
            total_fx = total_fx + seg_fx[k];
            total_fy = total_fy + seg_fy[k];
            total_mz = total_mz + seg_mz[k];
            total_fn = total_fn + seg_fn[k];
            total_power = total_power + seg_power[k];
            total_fn_x = total_fn_x + seg_fn_x[k];
            total_lat_x = total_lat_x + seg_lat_x[k];
            total_available = total_available + seg_available[k];
            total_consumed = total_consumed + seg_consumed[k];
            rolling_force = rolling_force + seg_rolling[k];
            total_world = total_world + vec3<f32>(seg_world_x[k], seg_world_y[k], seg_world_z[k]);
            if (seg_count[k] > 0u) {
                min_x = min(min_x, seg_min_x[k]);
                max_x = max(max_x, seg_max_x[k]);
                total_count = total_count + seg_count[k];
                total_slide = total_slide + seg_slide[k];
            }
            k = k + 1u;
        }
        let longitudinal_force = total_fx - rolling_force;
        let patch_length = max(max_x - min_x, 0.0);
        var contact_center_x: f32 = 0.0;
        if (total_fn > 1.0e-9) {
            contact_center_x = total_fn_x / total_fn;
        }
        var force_resultant_x: f32 = contact_center_x;
        if (abs(total_fy) > 1.0e-9) {
            force_resultant_x = total_lat_x / abs(total_fy);
        }
        let pneumatic_trail = contact_center_x - force_resultant_x;
        var sliding_fraction: f32 = 0.0;
        if (total_count > 0u) {
            sliding_fraction = f32(total_slide) / f32(total_count);
        }
        let adhesion_fraction = clamp(1.0 - sliding_fraction, 0.0, 1.0);
        var k_long: f32 = 0.0;
        let long_disp = abs(slip_ratio * radius);
        if (long_disp > 1.0e-6) {
            k_long = abs(total_fx) / long_disp;
        }
        var k_corner: f32 = 0.0;
        if (abs(slip_angle) > 1.0e-5) {
            k_corner = abs(total_fy) / abs(slip_angle);
        }
        outputs[wi].forces = vec4<f32>(total_fx, total_fn, total_fy, longitudinal_force);
        outputs[wi].diag1 = vec4<f32>(patch_length, slip_ratio, slip_speed, f32(total_count));
        outputs[wi].diag2 = vec4<f32>(f32(total_slide), total_mz, lateral_slip_speed, slip_angle);
        outputs[wi].diag3 = vec4<f32>(total_power, camber_angle, contact_center_x, pneumatic_trail);
        outputs[wi].diag4 = vec4<f32>(adhesion_fraction, sliding_fraction, total_available, total_consumed);
        outputs[wi].diag5 = vec4<f32>(k_long, k_corner, 0.0, 0.0);
        outputs[wi].world_force = vec4<f32>(total_world, 0.0);
    }
}
