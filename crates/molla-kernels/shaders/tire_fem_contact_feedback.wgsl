// GPU FEM tread contact feedback.
//
// One invocation owns one FEM node. If the node is a tread node inside the
// flat contact band, it advances that tread node's persistent bristle state,
// computes a smooth Coulomb-capped brush force, and feeds the resulting impulse
// back into the node velocity. This is the first GPU counterpart to
// `TetraTireFem::apply_tread_contact_wrench`.

struct Uniforms {
    // x=node_count, y=tread_count, z=sdf primitive count, w=triangle count
    counts: vec4<u32>,
    // x=dt, y=contact_band, z=ground_y, w=omega
    phys0: vec4<f32>,
    // xyz=hub position, w=rolling radius
    hub: vec4<f32>,
    // xyz=hub/world velocity, w=normal_load_per_tread_node
    hub_vel: vec4<f32>,
    // xyz=wheel heading, w=area_per_tread_node
    heading: vec4<f32>,
    // x=tread_shear_stiffness, y=mu_static, z=mu_dynamic, w=bristle_damping
    material0: vec4<f32>,
    // x=cohesion, y=bristle_relax_rate, zw=reserved
    material1: vec4<f32>,
    // xyz=terrain surface velocity, w=reserved
    terrain_velocity: vec4<f32>,
    // x=origin_x, y=origin_z, z=spacing, w=reference friction
    terrain_origin: vec4<f32>,
    // x=nx, y=nz, z=heightfield_enabled, w=material flags (bit0 friction, bit1 water film, bit2 sinkage, bit3 velocity)
    terrain_dims: vec4<u32>,
    // x=soil rolling-resistance scale, yzw=reserved
    soil0: vec4<f32>,
};

@group(0) @binding(0) var<storage, read>       pos_in:      array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       vel_in:      array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>       mass_in:     array<f32>;
@group(0) @binding(3) var<storage, read>       tread_ids:   array<u32>;
@group(0) @binding(4) var<storage, read>       bristle_in:  array<vec4<f32>>;

@group(0) @binding(5) var<storage, read_write> vel_out:     array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> bristle_out: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> force_out:   array<vec4<f32>>;
@group(0) @binding(8) var<storage, read>       terrain_heights:  array<f32>;
@group(0) @binding(9) var<storage, read>       terrain_friction: array<f32>;
@group(0) @binding(10) var<storage, read>      terrain_water_film: array<f32>;
@group(0) @binding(11) var<storage, read>      contact_primitives: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read>      terrain_sinkage: array<f32>;
@group(0) @binding(13) var<storage, read>      terrain_velocity_map: array<vec4<f32>>;

@group(1) @binding(0) var<uniform> u: Uniforms;

struct ContactSample {
    normal: vec3<f32>,
    distance: f32,
    friction: f32,
    water_film: f32,
    sinkage: f32,
};

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

fn sample_terrain_height(world_x: f32, world_z: f32) -> f32 {
    if (u.terrain_dims.z == 0u) {
        return u.phys0.z;
    }
    let nx = u.terrain_dims.x;
    let nz = u.terrain_dims.y;
    let spacing = max(u.terrain_origin.z, 1.0e-6);
    let fx = (world_x - u.terrain_origin.x) / spacing;
    let fz = (world_z - u.terrain_origin.y) / spacing;
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

fn sample_terrain_friction(world_x: f32, world_z: f32) -> f32 {
    let ref_mu = max(u.terrain_origin.w, 0.0);
    if (u.terrain_dims.z == 0u || (u.terrain_dims.w & 1u) == 0u) {
        return ref_mu;
    }
    let nx = u.terrain_dims.x;
    let nz = u.terrain_dims.y;
    let spacing = max(u.terrain_origin.z, 1.0e-6);
    let fx = (world_x - u.terrain_origin.x) / spacing;
    let fz = (world_z - u.terrain_origin.y) / spacing;
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
    return f0 * (1.0 - tz) + f1 * tz;
}

fn sample_terrain_water_film(world_x: f32, world_z: f32) -> f32 {
    if (u.terrain_dims.z == 0u || (u.terrain_dims.w & 2u) == 0u) {
        return 0.0;
    }
    let nx = u.terrain_dims.x;
    let nz = u.terrain_dims.y;
    let spacing = max(u.terrain_origin.z, 1.0e-6);
    let fx = (world_x - u.terrain_origin.x) / spacing;
    let fz = (world_z - u.terrain_origin.y) / spacing;
    let ix0_i = i32(floor(fx));
    let iz0_i = i32(floor(fz));
    let tx = fx - f32(ix0_i);
    let tz = fz - f32(iz0_i);
    let ix0 = u32(clamp(ix0_i, 0, i32(nx) - 1));
    let iz0 = u32(clamp(iz0_i, 0, i32(nz) - 1));
    let ix1 = u32(clamp(ix0_i + 1, 0, i32(nx) - 1));
    let iz1 = u32(clamp(iz0_i + 1, 0, i32(nz) - 1));
    let f00 = max(terrain_water_film[iz0 * nx + ix0], 0.0);
    let f10 = max(terrain_water_film[iz0 * nx + ix1], 0.0);
    let f01 = max(terrain_water_film[iz1 * nx + ix0], 0.0);
    let f11 = max(terrain_water_film[iz1 * nx + ix1], 0.0);
    let f0 = f00 * (1.0 - tx) + f10 * tx;
    let f1 = f01 * (1.0 - tx) + f11 * tx;
    return f0 * (1.0 - tz) + f1 * tz;
}

fn sample_terrain_sinkage(world_x: f32, world_z: f32) -> f32 {
    if (u.terrain_dims.z == 0u || (u.terrain_dims.w & 4u) == 0u) {
        return 0.0;
    }
    let nx = u.terrain_dims.x;
    let nz = u.terrain_dims.y;
    let spacing = max(u.terrain_origin.z, 1.0e-6);
    let fx = (world_x - u.terrain_origin.x) / spacing;
    let fz = (world_z - u.terrain_origin.y) / spacing;
    let ix0_i = i32(floor(fx));
    let iz0_i = i32(floor(fz));
    let tx = fx - f32(ix0_i);
    let tz = fz - f32(iz0_i);
    let ix0 = u32(clamp(ix0_i, 0, i32(nx) - 1));
    let iz0 = u32(clamp(iz0_i, 0, i32(nz) - 1));
    let ix1 = u32(clamp(ix0_i + 1, 0, i32(nx) - 1));
    let iz1 = u32(clamp(iz0_i + 1, 0, i32(nz) - 1));
    let s00 = max(terrain_sinkage[iz0 * nx + ix0], 0.0);
    let s10 = max(terrain_sinkage[iz0 * nx + ix1], 0.0);
    let s01 = max(terrain_sinkage[iz1 * nx + ix0], 0.0);
    let s11 = max(terrain_sinkage[iz1 * nx + ix1], 0.0);
    let s0 = s00 * (1.0 - tx) + s10 * tx;
    let s1 = s01 * (1.0 - tx) + s11 * tx;
    return s0 * (1.0 - tz) + s1 * tz;
}

fn sample_terrain_surface_velocity(world_x: f32, world_z: f32) -> vec3<f32> {
    if (u.terrain_dims.z == 0u || (u.terrain_dims.w & 8u) == 0u) {
        return u.terrain_velocity.xyz;
    }
    let nx = u.terrain_dims.x;
    let nz = u.terrain_dims.y;
    let spacing = max(u.terrain_origin.z, 1.0e-6);
    let fx = (world_x - u.terrain_origin.x) / spacing;
    let fz = (world_z - u.terrain_origin.y) / spacing;
    let ix0_i = i32(floor(fx));
    let iz0_i = i32(floor(fz));
    let tx = fx - f32(ix0_i);
    let tz = fz - f32(iz0_i);
    let ix0 = u32(clamp(ix0_i, 0, i32(nx) - 1));
    let iz0 = u32(clamp(iz0_i, 0, i32(nz) - 1));
    let ix1 = u32(clamp(ix0_i + 1, 0, i32(nx) - 1));
    let iz1 = u32(clamp(iz0_i + 1, 0, i32(nz) - 1));
    let v00 = terrain_velocity_map[iz0 * nx + ix0].xyz;
    let v10 = terrain_velocity_map[iz0 * nx + ix1].xyz;
    let v01 = terrain_velocity_map[iz1 * nx + ix0].xyz;
    let v11 = terrain_velocity_map[iz1 * nx + ix1].xyz;
    let v0 = v00 * (1.0 - tx) + v10 * tx;
    let v1 = v01 * (1.0 - tx) + v11 * tx;
    return v0 * (1.0 - tz) + v1 * tz;
}

fn wet_film_factors(water_depth_m: f32, speed_mps: f32) -> vec4<f32> {
    let water = max(water_depth_m, 0.0);
    if (water <= 1.0e-7) {
        return vec4<f32>(1.0);
    }
    // Mirrors the CPU default SurfaceFilmState with fresh tread. The GPU
    // feedback pass has no per-node wear state yet, so tread-depth effects
    // stay on the CPU/readout path until a tread-wear buffer is added.
    let void_depth = 0.004;
    let drainage_speed = 18.0;
    let film_ratio = clamp(water / (void_depth + 2.0e-4), 0.0, 8.0);
    let speed_ratio = max(speed_mps, 0.0) / max(drainage_speed, 0.5);
    let severity = film_ratio * (1.0 + 0.85 * pow(speed_ratio, 1.35));
    let static_factor = clamp(1.0 / (1.0 + 0.48 * severity), 0.12, 1.0);
    let dynamic_factor = clamp(1.0 / (1.0 + 0.68 * severity), 0.08, 1.0);
    let damping_factor = clamp(1.0 / (1.0 + 0.20 * severity), 0.35, 1.0);
    let cohesion_factor = clamp(1.0 - 0.22 * severity, 0.0, 1.0);
    return vec4<f32>(static_factor, min(dynamic_factor, static_factor), damping_factor, cohesion_factor);
}

fn sample_terrain_normal(world_x: f32, world_z: f32) -> vec3<f32> {
    if (u.terrain_dims.z == 0u) {
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    let d = max(u.terrain_origin.z * 0.5, 1.0e-5);
    let dhdx = (sample_terrain_height(world_x + d, world_z) - sample_terrain_height(world_x - d, world_z)) / (2.0 * d);
    let dhdz = (sample_terrain_height(world_x, world_z + d) - sample_terrain_height(world_x, world_z - d)) / (2.0 * d);
    return safe_normalize(vec3<f32>(-dhdx, 1.0, -dhdz), vec3<f32>(0.0, 1.0, 0.0));
}

fn signed_axis(v: f32) -> f32 {
    if (v >= 0.0) {
        return 1.0;
    }
    return -1.0;
}

fn sphere_contact(p: vec3<f32>, row0: vec4<f32>, row1: vec4<f32>) -> ContactSample {
    let center = row0.xyz;
    let radius = max(row1.x, 1.0e-6);
    let delta = p - center;
    let len_delta = length(delta);
    let normal = safe_normalize(delta, vec3<f32>(0.0, 1.0, 0.0));
    return ContactSample(normal, len_delta - radius, max(row1.w, 0.0), 0.0, 0.0);
}

fn box_contact(p: vec3<f32>, row0: vec4<f32>, row1: vec4<f32>) -> ContactSample {
    let center = row0.xyz;
    let half_extents = max(row1.xyz, vec3<f32>(1.0e-6));
    let local = p - center;
    let q = abs(local) - half_extents;
    let outside = max(q, vec3<f32>(0.0));
    let outside_len = length(outside);
    var normal = vec3<f32>(0.0, 1.0, 0.0);
    if (outside_len > 1.0e-9) {
        normal = safe_normalize(
            vec3<f32>(
                signed_axis(local.x) * outside.x,
                signed_axis(local.y) * outside.y,
                signed_axis(local.z) * outside.z,
            ),
            vec3<f32>(0.0, 1.0, 0.0),
        );
    } else if (q.x >= q.y && q.x >= q.z) {
        normal = vec3<f32>(signed_axis(local.x), 0.0, 0.0);
    } else if (q.y >= q.z) {
        normal = vec3<f32>(0.0, signed_axis(local.y), 0.0);
    } else {
        normal = vec3<f32>(0.0, 0.0, signed_axis(local.z));
    }
    let inside = min(max(q.x, max(q.y, q.z)), 0.0);
    return ContactSample(normal, outside_len + inside, max(row1.w, 0.0), 0.0, 0.0);
}

fn triangle_normal(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    var n = cross(b - a, c - a);
    if (n.y < 0.0) {
        n = -n;
    }
    return safe_normalize(n, vec3<f32>(0.0, 1.0, 0.0));
}

fn closest_point_on_triangle(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) {
        return a;
    }

    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) {
        return b;
    }

    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / max(d1 - d3, 1.0e-12);
        return a + ab * v;
    }

    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / max(d2 - d6, 1.0e-12);
        return a + ac * w;
    }

    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / max((d4 - d3) + (d5 - d6), 1.0e-12);
        return b + (c - b) * w;
    }

    let denom = max(va + vb + vc, 1.0e-12);
    let v = vb / denom;
    let w = vc / denom;
    return a + ab * v + ac * w;
}

fn triangle_contact(p: vec3<f32>, row0: vec4<f32>, row1: vec4<f32>, row2: vec4<f32>, row3: vec4<f32>) -> ContactSample {
    let a = row0.xyz;
    let b = row1.xyz;
    let c = row2.xyz;
    let normal = triangle_normal(a, b, c);
    let closest = closest_point_on_triangle(p, a, b, c);
    let delta = p - closest;
    let unsigned_distance = length(delta);
    let signed_distance = select(-unsigned_distance, unsigned_distance, dot(delta, normal) >= 0.0);
    return ContactSample(normal, signed_distance, max(row3.x, 0.0), 0.0, 0.0);
}

fn triangle_broadphase_reject(p: vec3<f32>, row3: vec4<f32>, margin: f32) -> bool {
    let center = row3.yz;
    let radius = max(row3.w, 0.0) + max(margin, 0.0);
    let dxz = vec2<f32>(p.x, p.z) - center;
    return dot(dxz, dxz) > radius * radius;
}

fn triangle_bvh_aabb_distance2(p: vec3<f32>, node_row: u32) -> f32 {
    let row0 = contact_primitives[node_row];
    let row1 = contact_primitives[node_row + 1u];
    let mn = vec3<f32>(row0.x, row0.y, row0.z);
    let mx = vec3<f32>(row0.w, row1.x, row1.y);
    let below = max(mn - p, vec3<f32>(0.0));
    let above = max(p - mx, vec3<f32>(0.0));
    let d = below + above;
    return dot(d, d);
}

fn triangle_bvh_read_index(index_base: u32, index_id: u32) -> u32 {
    let row = contact_primitives[index_base + index_id / 4u];
    let lane = index_id % 4u;
    if (lane == 0u) {
        return u32(row.x);
    }
    if (lane == 1u) {
        return u32(row.y);
    }
    if (lane == 2u) {
        return u32(row.z);
    }
    return u32(row.w);
}

fn sample_contact(p: vec3<f32>) -> ContactSample {
    let h = sample_terrain_height(p.x, p.z);
    var best = ContactSample(
        sample_terrain_normal(p.x, p.z),
        p.y - h,
        sample_terrain_friction(p.x, p.z),
        sample_terrain_water_film(p.x, p.z),
        sample_terrain_sinkage(p.x, p.z),
    );
    for (var i: u32 = 0u; i < u.counts.z; i = i + 1u) {
        let row0 = contact_primitives[i * 2u];
        let row1 = contact_primitives[i * 2u + 1u];
        var candidate: ContactSample;
        if (row0.w < 1.5) {
            candidate = sphere_contact(p, row0, row1);
        } else {
            candidate = box_contact(p, row0, row1);
        }
        if (candidate.distance < best.distance) {
            best = candidate;
        }
    }
    let tri_base = u.counts.z * 2u;
    let tri_margin = max(u.phys0.y, u.terrain_origin.z * 0.5);
    var tri_weight_sum = 0.0;
    var tri_normal_sum = vec3<f32>(0.0);
    var tri_friction_sum = 0.0;
    var tri_min_distance = 3.4028235e38;
    let bvh_count = u32(max(u.soil0.y, 0.0));
    if (bvh_count > 0u) {
        let bvh_base = tri_base + u.counts.w * 4u;
        let index_base = bvh_base + bvh_count * 3u;
        var stack: array<u32, 64>;
        var stack_len: u32 = 1u;
        stack[0] = 0u;
        loop {
            if (stack_len == 0u) {
                break;
            }
            stack_len = stack_len - 1u;
            let node_id = stack[stack_len];
            if (node_id >= bvh_count) {
                continue;
            }
            let node_row = bvh_base + node_id * 3u;
            let search_radius = abs(best.distance) + tri_margin;
            if (triangle_bvh_aabb_distance2(p, node_row) > search_radius * search_radius) {
                continue;
            }
            let row1 = contact_primitives[node_row + 1u];
            let row2 = contact_primitives[node_row + 2u];
            if (row1.z > 0.5) {
                let start = u32(row1.w);
                let count = u32(row2.x);
                for (var j: u32 = 0u; j < count; j = j + 1u) {
                    let tri_id = triangle_bvh_read_index(index_base, start + j);
                    if (tri_id >= u.counts.w) {
                        continue;
                    }
                    let row = tri_base + tri_id * 4u;
                    if (triangle_broadphase_reject(p, contact_primitives[row + 3u], tri_margin)) {
                        continue;
                    }
                    let candidate = triangle_contact(
                        p,
                        contact_primitives[row],
                        contact_primitives[row + 1u],
                        contact_primitives[row + 2u],
                        contact_primitives[row + 3u],
                    );
                    let manifold_penetration = tri_margin - candidate.distance;
                    if (manifold_penetration > 0.0) {
                        let w = max(manifold_penetration, 1.0e-6);
                        tri_weight_sum += w;
                        tri_normal_sum += candidate.normal * w;
                        tri_friction_sum += candidate.friction * w;
                        tri_min_distance = min(tri_min_distance, candidate.distance - tri_margin);
                    } else if (candidate.distance < best.distance) {
                        best = candidate;
                    }
                }
            } else {
                let left = u32(row1.w);
                let right = u32(row2.x);
                if (right < bvh_count && stack_len < 64u) {
                    stack[stack_len] = right;
                    stack_len = stack_len + 1u;
                }
                if (left < bvh_count && stack_len < 64u) {
                    stack[stack_len] = left;
                    stack_len = stack_len + 1u;
                }
            }
        }
    } else {
        for (var i: u32 = 0u; i < u.counts.w; i = i + 1u) {
            let row = tri_base + i * 4u;
            if (triangle_broadphase_reject(p, contact_primitives[row + 3u], tri_margin)) {
                continue;
            }
            let candidate = triangle_contact(
                p,
                contact_primitives[row],
                contact_primitives[row + 1u],
                contact_primitives[row + 2u],
                contact_primitives[row + 3u],
            );
            let manifold_penetration = tri_margin - candidate.distance;
            if (manifold_penetration > 0.0) {
                let w = max(manifold_penetration, 1.0e-6);
                tri_weight_sum += w;
                tri_normal_sum += candidate.normal * w;
                tri_friction_sum += candidate.friction * w;
                tri_min_distance = min(tri_min_distance, candidate.distance - tri_margin);
            } else if (candidate.distance < best.distance) {
                best = candidate;
            }
        }
    }
    if (tri_weight_sum > 0.0) {
        let tri_sample = ContactSample(
            safe_normalize(tri_normal_sum, best.normal),
            tri_min_distance,
            tri_friction_sum / tri_weight_sum,
            best.water_film,
            best.sinkage,
        );
        if (tri_sample.distance < best.distance) {
            best = tri_sample;
        }
    }
    return best;
}

@compute @workgroup_size(64)
fn contact_feedback(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.counts.x) {
        return;
    }

    var v_next = vel_in[i].xyz;
    var force = vec3<f32>(0.0);
    var tread_slot: u32 = 0xffffffffu;
    for (var t: u32 = 0u; t < u.counts.y; t = t + 1u) {
        if (tread_ids[t] == i) {
            tread_slot = t;
        }
    }

    if (tread_slot != 0xffffffffu) {
        let p = pos_in[i].xyz;
        let contact = sample_contact(p);
        let normal = contact.normal;
        let clearance = contact.distance;
        let band = max(u.phys0.y, 1.0e-6);
        let dt = max(u.phys0.x, 0.0);
        if (clearance <= band && dt > 0.0) {
            let weight = clamp((band - clearance) / band, 0.0, 3.0);
            let normal_load = max(u.hub_vel.w, 0.0) * weight;
            let heading = safe_normalize(
                u.heading.xyz - normal * dot(u.heading.xyz, normal),
                vec3<f32>(1.0, 0.0, 0.0),
            );
            let side = safe_normalize(cross(heading, normal), vec3<f32>(0.0, 0.0, 1.0));
            let terrain_velocity = sample_terrain_surface_velocity(p.x, p.z);
            let rel_v = v_next - terrain_velocity;
            let v_long = dot(rel_v, heading);
            let v_lat = dot(rel_v, side);
            let radius = max(u.hub.w, 1.0e-4);
            let wheel_speed = u.phys0.w * radius;
            let slip_speed = wheel_speed - v_long;
            let lateral_slip_speed = -v_lat;

            var b_long = dot(bristle_in[tread_slot].xyz, heading) + slip_speed * dt;
            var b_lat = dot(bristle_in[tread_slot].xyz, side) + lateral_slip_speed * dt;
            let area = max(u.heading.w, 1.0e-8);
            let shear_k = max(u.material0.x, 0.0);
            let water = contact.water_film;
            let wet = wet_film_factors(water, sqrt(slip_speed * slip_speed + lateral_slip_speed * lateral_slip_speed));
            let damping = max(u.material0.w, 0.0) * wet.z;
            let trial_long = shear_k * area * b_long + damping * area * slip_speed;
            let trial_lat = shear_k * area * b_lat + damping * area * lateral_slip_speed;
            let terrain_mu_ref = max(u.terrain_origin.w, max(u.material0.y, 1.0e-3));
            let terrain_mu = contact.friction;
            let friction_scale = clamp(terrain_mu / max(terrain_mu_ref, 1.0e-6), 0.0, 3.0);
            let cohesion = max(u.material1.x, 0.0) * wet.w;
            let static_cap = max(u.material0.y, 0.0) * wet.x * friction_scale * normal_load + cohesion * area;
            let dynamic_cap = max(u.material0.z, 0.0) * wet.y * friction_scale * normal_load + cohesion * area;
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
            let rolling_resistance = max(u.soil0.x, 0.0) * normal_load * max(contact.sinkage, 0.0) / max(radius, 1.0e-4);
            if (rolling_resistance > 0.0) {
                let rr_sign = select(1.0, -1.0, v_long >= 0.0);
                force = force + heading * (rr_sign * rolling_resistance);
            }
            let inv_m = 1.0 / max(mass_in[i], 1.0e-9);
            v_next = clamp(v_next + force * inv_m * dt, vec3<f32>(-12.0), vec3<f32>(12.0));
        } else {
            let decay = exp(-max(u.material1.y, 0.0) * dt);
            bristle_out[tread_slot] = vec4<f32>(bristle_in[tread_slot].xyz * decay, 0.0);
        }
    }

    vel_out[i] = vec4<f32>(v_next, 0.0);
    force_out[i] = vec4<f32>(force, 0.0);
}
