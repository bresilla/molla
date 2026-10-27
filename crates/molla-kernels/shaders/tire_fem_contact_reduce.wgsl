// GPU FEM tire contact diagnostics reduction.
//
// One workgroup reduces the current deformed tread nodes into the same coarse
// patch diagnostics the CPU FEM contact path uses for viewer/readout work.
// Supports flat ground or regular heightfield terrain with optional friction
// samples. Persistent GPU bristles/force feedback are follow-up kernels.

struct Uniforms {
    // x=node_count, y=tread_count, z=sdf primitive count, w=triangle count
    counts: vec4<u32>,
    // x=contact_band, y=ground_y, z=downforce, w=reserved
    contact0: vec4<f32>,
    // xyz=hub, w=reserved
    hub: vec4<f32>,
    // xyz=heading, w=reserved
    heading: vec4<f32>,
    // x=origin_x, y=origin_z, z=spacing, w=reference friction
    terrain_origin: vec4<f32>,
    // x=nx, y=nz, z=heightfield_enabled, w=material flags (bit0 friction, bit1 water film, bit2 sinkage)
    terrain_dims: vec4<u32>,
    // x=soil rolling-resistance scale, yzw=reserved
    soil0: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> tread_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_diag: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> terrain_heights: array<f32>;
@group(0) @binding(4) var<storage, read> terrain_friction: array<f32>;
@group(0) @binding(5) var<storage, read> terrain_water_film: array<f32>;
@group(0) @binding(6) var<storage, read> contact_primitives: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read> terrain_sinkage: array<f32>;
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

fn sample_terrain_height(world_x: f32, world_z: f32) -> f32 {
    if (u.terrain_dims.z == 0u) {
        return u.contact0.y;
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
    let tri_margin = max(u.contact0.x, u.terrain_origin.z * 0.5);
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
fn reduce_contact(@builtin(local_invocation_id) lid: vec3<u32>) {
    if (lid.x != 0u) { return; }

    let n_nodes = u.counts.x;
    let n_tread = u.counts.y;
    let band = max(u.contact0.x, 1.0e-6);
    let downforce = max(u.contact0.z, 0.0);
    let hub = u.hub.xyz;
    let heading = safe_normalize(u.heading.xyz, vec3<f32>(1.0, 0.0, 0.0));

    var weight_sum = 0.0;
    var contact_count = 0.0;
    var min_x = 3.4028235e38;
    var max_x = -3.4028235e38;

    for (var i: u32 = 0u; i < n_tread; i = i + 1u) {
        let id = tread_ids[i];
        if (id >= n_nodes) { continue; }
        let p = positions[id].xyz;
        let contact = sample_contact(p);
        let clearance = contact.distance;
        if (clearance > band) { continue; }
        let w = clamp((band - clearance) / band, 0.0, 3.0);
        if (w <= 0.0) { continue; }
        let lx = dot(p - hub, heading);
        weight_sum = weight_sum + w;
        contact_count = contact_count + 1.0;
        min_x = min(min_x, lx);
        max_x = max(max_x, lx);
    }

    if (weight_sum <= 1.0e-12 || downforce <= 0.0 || contact_count <= 0.0) {
        out_diag[0] = vec4<f32>(0.0);
        out_diag[1] = vec4<f32>(0.0);
        out_diag[2] = vec4<f32>(0.0);
        out_diag[3] = vec4<f32>(0.0);
        return;
    }

    var normal_load = 0.0;
    var moment_x = 0.0;
    var friction_weighted = 0.0;
    var water_weighted = 0.0;
    var sinkage_weighted = 0.0;
    var world_force = vec3<f32>(0.0);
    var world_moment = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < n_tread; i = i + 1u) {
        let id = tread_ids[i];
        if (id >= n_nodes) { continue; }
        let p = positions[id].xyz;
        let contact = sample_contact(p);
        let clearance = contact.distance;
        if (clearance > band) { continue; }
        let w = clamp((band - clearance) / band, 0.0, 3.0);
        if (w <= 0.0) { continue; }
        let lx = dot(p - hub, heading);
        let fn_i = downforce * w / weight_sum;
        normal_load = normal_load + fn_i;
        moment_x = moment_x + lx * fn_i;
        friction_weighted = friction_weighted + contact.friction * w;
        water_weighted = water_weighted + contact.water_film * w;
        sinkage_weighted = sinkage_weighted + contact.sinkage * w;
        let force_i = contact.normal * fn_i;
        world_force = world_force + force_i;
        world_moment = world_moment + cross(p - hub, force_i);
    }

    let patch_length = max(max_x - min_x, 0.0);
    let center_x = moment_x / max(normal_load, 1.0e-9);
    let mean_friction = friction_weighted / max(weight_sum, 1.0e-9);
    let mean_water = water_weighted / max(weight_sum, 1.0e-9);
    let mean_sinkage = sinkage_weighted / max(weight_sum, 1.0e-9);
    out_diag[0] = vec4<f32>(normal_load, contact_count, patch_length, center_x);
    out_diag[1] = vec4<f32>(min_x, max_x, weight_sum, mean_friction);
    out_diag[2] = vec4<f32>(world_force, mean_sinkage);
    out_diag[3] = vec4<f32>(world_moment, mean_water);
}
