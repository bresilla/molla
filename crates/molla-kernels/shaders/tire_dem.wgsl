struct ContactCell {
    center_radius: vec4<f32>,
    force: vec4<f32>,
};

struct DemUniforms {
    dims: vec4<u32>,
    // x: grain_mass, y: grain_radius, z: dt, w: tire_position_push_scale
    params0: vec4<f32>,
    // x: normal_stiffness, y: damping, z: friction, w: floor_y
    params1: vec4<f32>,
    gravity: vec4<f32>,
    // x/y/z: grid origin, w: cell size
    grid_origin_cell_size: vec4<f32>,
    // x/y/z: grid shape
    grid_shape: vec4<u32>,
};

@group(0) @binding(0) var<storage, read> pos_in: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> vel_in: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> contacts: array<ContactCell>;
@group(0) @binding(3) var<storage, read_write> pos_out: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> vel_out: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> sorted_indices: array<u32>;
@group(0) @binding(6) var<storage, read_write> cell_ranges: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read_write> gpu_cell_counts: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> gpu_cell_slots: array<u32>;
@group(0) @binding(9) var<storage, read_write> sorted_cell_keys: array<u32>;
@group(0) @binding(10) var<storage, read> lbvh_nodes: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read> lbvh_sorted_indices: array<u32>;
@group(0) @binding(12) var<storage, read> terrain_reference_heights: array<f32>;
@group(0) @binding(13) var<storage, read_write> terrain_heights: array<f32>;
@group(0) @binding(14) var<storage, read_write> terrain_sinkage: array<f32>;
@group(0) @binding(15) var<storage, read_write> terrain_surface_velocity: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> u: DemUniforms;

fn tire_delta(p: vec4<f32>, mass: f32, radius: f32, dt: f32, push_scale: f32) -> vec4<f32> {
    var dv = vec3<f32>(0.0, 0.0, 0.0);
    var push_y = 0.0;
    var c = 0u;
    loop {
        if (c >= u.dims.y) {
            break;
        }
        let contact = contacts[c];
        let cx = contact.center_radius.x;
        let surface_y = contact.center_radius.y;
        let cz = contact.center_radius.z;
        let influence = max(contact.center_radius.w, radius * 1.25);
        let dx = p.x - cx;
        let dz = p.z - cz;
        let horizontal = sqrt(dx * dx + dz * dz);
        if (horizontal <= influence + radius) {
            let below_surface = (surface_y + radius * 1.5) - p.y;
            if (below_surface >= -radius && below_surface <= radius * 5.0) {
                let radial_w = clamp(1.0 - horizontal / (influence + radius), 0.0, 1.0);
                let depth_w = clamp(1.0 - max(below_surface, 0.0) / (radius * 5.0), 0.15, 1.0);
                let w = radial_w * radial_w * depth_w;
                dv += contact.force.xyz * (w * dt / mass);
                if (contact.force.y < 0.0) {
                    push_y += (-contact.force.y) * w * dt * dt / mass * push_scale;
                }
            }
        }
        c += 1u;
    }
    return vec4<f32>(dv, min(push_y, radius * 0.35));
}

fn pair_force(
    pa: vec3<f32>,
    va: vec3<f32>,
    pb: vec3<f32>,
    vb: vec3<f32>,
    sum_radii: f32,
    normal_stiffness: f32,
    damping: f32,
    friction: f32
) -> vec3<f32> {
    let d = pa - pb;
    let dist = length(d);
    let overlap = sum_radii - dist;
    if (overlap <= 0.0 || dist < 1.0e-9) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let n = d / dist;
    let v_rel = va - vb;
    let v_n = dot(v_rel, n);
    let fn_mag = max(normal_stiffness * overlap - damping * v_n, 0.0);
    let v_t = v_rel - n * v_n;
    let vt = length(v_t);
    var f = n * fn_mag;
    if (vt > 1.0e-9) {
        let ft = max(friction, 0.0) * fn_mag * (vt / (vt + 0.05));
        f -= v_t * (ft / vt);
    }
    return f;
}

fn cell_coords(p: vec3<f32>) -> vec3<u32> {
    let raw = floor((p - u.grid_origin_cell_size.xyz) / max(u.grid_origin_cell_size.w, 1.0e-9));
    let max_shape = vec3<i32>(
        i32(max(u.grid_shape.x, 1u)) - 1,
        i32(max(u.grid_shape.y, 1u)) - 1,
        i32(max(u.grid_shape.z, 1u)) - 1
    );
    let clamped = clamp(vec3<i32>(raw), vec3<i32>(0, 0, 0), max_shape);
    return vec3<u32>(clamped);
}

fn cell_id(c: vec3<u32>) -> u32 {
    return c.x + c.y * u.grid_shape.x + c.z * u.grid_shape.x * u.grid_shape.y;
}

fn expand_bits_10(v_in: u32) -> u32 {
    var v = v_in & 0x3ffu;
    v = (v | (v << 16u)) & 0x030000ffu;
    v = (v | (v << 8u)) & 0x0300f00fu;
    v = (v | (v << 4u)) & 0x030c30c3u;
    v = (v | (v << 2u)) & 0x09249249u;
    return v;
}

fn morton3_10(x: u32, y: u32, z: u32) -> u32 {
    return expand_bits_10(x) | (expand_bits_10(y) << 1u) | (expand_bits_10(z) << 2u);
}

fn aabb_distance2_to_point(mn: vec3<f32>, mx: vec3<f32>, p: vec3<f32>) -> f32 {
    let below = max(mn - p, vec3<f32>(0.0));
    let above = max(p - mx, vec3<f32>(0.0));
    let d = below + above;
    return dot(d, d);
}

@compute @workgroup_size(64)
fn build_unbounded_cell_keys(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.dims.x) {
        return;
    }
    sorted_cell_keys[i] = cell_id(cell_coords(pos_in[i].xyz));
    sorted_indices[i] = i;
}

@compute @workgroup_size(64)
fn build_sparse_lbvh_morton_keys(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.dims.x) {
        return;
    }
    let scene_min = u.grid_origin_cell_size.xyz;
    let extent = max(u.params1.xyz, vec3<f32>(1.0e-9));
    let normalized = clamp((pos_in[i].xyz - scene_min) / extent, vec3<f32>(0.0), vec3<f32>(1.0));
    let q = vec3<u32>(round(normalized * 1023.0));
    sorted_cell_keys[i] = morton3_10(q.x, q.y, q.z);
    sorted_indices[i] = i;
}

@compute @workgroup_size(64)
fn build_sparse_lbvh_leaf_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let leaf_id = gid.x;
    let grain_count = u.dims.x;
    let leaf_count = u.dims.y;
    let leaf_size = max(u.dims.z, 1u);
    var active_leaf_count = u.grid_shape.w;
    if (active_leaf_count == 0u) {
        active_leaf_count = leaf_count;
    }
    if (leaf_id >= leaf_count || grain_count == 0u) {
        return;
    }

    let leaf_node_offset = u.grid_shape.x;
    let row = (leaf_node_offset + leaf_id) * 3u;
    if (leaf_id >= active_leaf_count) {
        pos_out[row] = vec4<f32>(3.4028234663852886e38, 3.4028234663852886e38, 3.4028234663852886e38, f32(0xffffffffu));
        pos_out[row + 1u] = vec4<f32>(-3.4028234663852886e38, -3.4028234663852886e38, -3.4028234663852886e38, f32(0xffffffffu));
        pos_out[row + 2u] = vec4<f32>(f32(grain_count), 0.0, 1.0, 0.0);
        return;
    }

    let start = leaf_id * leaf_size;
    let end = min(start + leaf_size, grain_count);
    if (start >= end) {
        return;
    }
    var first = sorted_indices[start];
    if (first >= grain_count) {
        first = 0u;
    }
    var mn = pos_in[first].xyz;
    var mx = mn;
    var cursor = start + 1u;
    loop {
        if (cursor >= end) {
            break;
        }
        let grain_id = sorted_indices[cursor];
        if (grain_id < grain_count) {
            let p = pos_in[grain_id].xyz;
            mn = min(mn, p);
            mx = max(mx, p);
        }
        cursor += 1u;
    }

    pos_out[row] = vec4<f32>(mn, f32(0xffffffffu));
    pos_out[row + 1u] = vec4<f32>(mx, f32(0xffffffffu));
    pos_out[row + 2u] = vec4<f32>(f32(start), f32(end - start), 1.0, 0.0);
}

@compute @workgroup_size(64)
fn build_sparse_lbvh_internal_level(@builtin(global_invocation_id) gid: vec3<u32>) {
    let parent_id = gid.x;
    let parent_count = u.dims.y;
    if (parent_id >= parent_count) {
        return;
    }

    let parent_start = u.grid_shape.x;
    let child_start = u.grid_shape.y;
    let node_count = u.grid_shape.z;
    let parent_node = parent_start + parent_id;
    let left = child_start + parent_id * 2u;
    let right = left + 1u;
    if (parent_node >= node_count || right >= node_count) {
        return;
    }

    let left_min = pos_out[left * 3u].xyz;
    let left_max = pos_out[left * 3u + 1u].xyz;
    let right_min = pos_out[right * 3u].xyz;
    let right_max = pos_out[right * 3u + 1u].xyz;
    pos_out[parent_node * 3u] = vec4<f32>(min(left_min, right_min), f32(left));
    pos_out[parent_node * 3u + 1u] = vec4<f32>(max(left_max, right_max), f32(right));
    pos_out[parent_node * 3u + 2u] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn scan_unbounded_cell_ranges(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    if (cell >= u.dims.z) {
        return;
    }

    var start = u.dims.x;
    var end = u.dims.x;
    var i = 0u;
    loop {
        if (i >= u.dims.x) {
            break;
        }
        let key = sorted_cell_keys[i];
        if (key == cell) {
            if (start == u.dims.x) {
                start = i;
            }
            end = i + 1u;
        } else if (start != u.dims.x && key > cell) {
            break;
        }
        i += 1u;
    }
    if (start == u.dims.x) {
        cell_ranges[cell] = vec4<u32>(0u, 0u, 0u, 0u);
    } else {
        cell_ranges[cell] = vec4<u32>(start, end, end - start, 0u);
    }
}

@compute @workgroup_size(64)
fn apply_impulses(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.dims.x) {
        return;
    }
    let grain_mass = max(u.params0.x, 1.0e-9);
    let grain_radius = max(u.params0.y, 1.0e-9);
    let dt = max(u.params0.z, 0.0);
    let push_scale = max(u.params0.w, 0.0);

    var p = pos_in[i];
    var v = vel_in[i];
    let delta = tire_delta(p, grain_mass, grain_radius, dt, push_scale);
    v.x += delta.x;
    v.y += delta.y;
    v.z += delta.z;
    p.x += v.x * dt;
    p.y += v.y * dt;
    p.z += v.z * dt;
    p.y -= delta.w;
    pos_out[i] = p;
    vel_out[i] = v;
}

@compute @workgroup_size(64)
fn step_dem_lbvh(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.dims.x) {
        return;
    }

    let grain_mass = max(u.params0.x, 1.0e-9);
    let grain_radius = max(u.params0.y, 1.0e-9);
    let dt = max(u.params0.z, 0.0);
    let push_scale = max(u.params0.w, 0.0);
    let normal_stiffness = max(u.params1.x, 0.0);
    let damping = max(u.params1.y, 0.0);
    let friction = max(u.params1.z, 0.0);
    let floor_y = u.params1.w;

    let p0 = pos_in[i];
    let v0 = vel_in[i];
    var force = u.gravity.xyz * grain_mass;
    let search_radius = grain_radius * 2.0 + 1.0e-6;
    let search_radius2 = search_radius * search_radius;

    if (u.dims.z > 0u) {
        var stack: array<u32, 64>;
        var stack_len = 1u;
        stack[0] = 0u;
        loop {
            if (stack_len == 0u) {
                break;
            }
            stack_len = stack_len - 1u;
            let node_id = stack[stack_len];
            if (node_id >= u.dims.z) {
                continue;
            }
            let row = node_id * 3u;
            let row0 = lbvh_nodes[row];
            let row1 = lbvh_nodes[row + 1u];
            let row2 = lbvh_nodes[row + 2u];
            if (aabb_distance2_to_point(row0.xyz, row1.xyz, p0.xyz) > search_radius2) {
                continue;
            }
            if (row2.z > 0.5) {
                let start = u32(row2.x);
                let count = u32(row2.y);
                for (var k: u32 = 0u; k < count; k = k + 1u) {
                    let cursor = start + k;
                    if (cursor >= u.dims.w) {
                        continue;
                    }
                    let j = lbvh_sorted_indices[cursor];
                    if (j != i && j < u.dims.x) {
                        force += pair_force(
                            p0.xyz,
                            v0.xyz,
                            pos_in[j].xyz,
                            vel_in[j].xyz,
                            grain_radius * 2.0,
                            normal_stiffness,
                            damping,
                            friction
                        );
                    }
                }
            } else {
                let left = u32(row0.w);
                let right = u32(row1.w);
                if (right < u.dims.z && stack_len < 64u) {
                    stack[stack_len] = right;
                    stack_len = stack_len + 1u;
                }
                if (left < u.dims.z && stack_len < 64u) {
                    stack[stack_len] = left;
                    stack_len = stack_len + 1u;
                }
            }
        }
    }

    let pen = (floor_y + grain_radius) - p0.y;
    if (pen > 0.0) {
        let fn_mag = max(normal_stiffness * pen - damping * v0.y, 0.0);
        force.y += fn_mag;
        let vt = length(vec2<f32>(v0.x, v0.z));
        if (vt > 1.0e-9) {
            let ft = max(friction, 0.0) * fn_mag * (vt / (vt + 0.05));
            force.x -= v0.x * ft / vt;
            force.z -= v0.z * ft / vt;
        }
    }

    var v = v0;
    v.x += force.x * dt / grain_mass;
    v.y += force.y * dt / grain_mass;
    v.z += force.z * dt / grain_mass;
    let delta = tire_delta(p0, grain_mass, grain_radius, dt, push_scale);
    v.x += delta.x;
    v.y += delta.y;
    v.z += delta.z;
    var p = p0;
    p.x += v.x * dt;
    p.y += v.y * dt;
    p.z += v.z * dt;
    p.y -= delta.w;
    pos_out[i] = p;
    vel_out[i] = v;
}

@compute @workgroup_size(64)
fn raster_dem_fem_terrain(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cell = gid.x;
    let nx = max(u.dims.y, 1u);
    let nz = max(u.dims.z, 1u);
    let cell_count = nx * nz;
    if (cell >= cell_count) {
        return;
    }

    let ix = cell % nx;
    let iz = cell / nx;
    let spacing = max(u.grid_origin_cell_size.w, 1.0e-9);
    let x0 = u.grid_origin_cell_size.x + f32(ix) * spacing;
    let x1 = x0 + spacing;
    let z0 = u.grid_origin_cell_size.y + f32(iz) * spacing;
    let z1 = z0 + spacing;
    let grain_radius = max(u.params0.y, 1.0e-9);
    let floor_y = u.params1.w;

    var height = floor_y;
    var top_y = -3.4028234663852886e38;
    for (var i: u32 = 0u; i < u.dims.x; i = i + 1u) {
        let p = pos_in[i];
        if (p.x >= x0 && p.x < x1 && p.z >= z0 && p.z < z1) {
            height = max(height, p.y + grain_radius);
            top_y = max(top_y, p.y);
        }
    }

    let reference = terrain_reference_heights[cell];
    terrain_heights[cell] = height;
    terrain_sinkage[cell] = max(reference - height, 0.0);

    var velocity_sum = vec3<f32>(0.0, 0.0, 0.0);
    var count = 0.0;
    let surface_band_m = max(grain_radius * 2.25, spacing * 0.25);
    if (top_y > -1.0e30) {
        for (var i: u32 = 0u; i < u.dims.x; i = i + 1u) {
            let p = pos_in[i];
            if (
                p.x >= x0 && p.x < x1 &&
                p.z >= z0 && p.z < z1 &&
                p.y >= top_y - surface_band_m
            ) {
                velocity_sum += vel_in[i].xyz;
                count += 1.0;
            }
        }
    }
    if (count > 0.0) {
        terrain_surface_velocity[cell] = vec4<f32>(velocity_sum / count, count);
    } else {
        terrain_surface_velocity[cell] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
}

@compute @workgroup_size(64)
fn build_dem_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.dims.x || u.dims.w == 0u) {
        return;
    }
    let cell = cell_id(cell_coords(pos_in[i].xyz));
    if (cell >= u.dims.z) {
        return;
    }
    let slot = atomicAdd(&gpu_cell_counts[cell], 1u);
    if (slot < u.dims.w) {
        gpu_cell_slots[cell * u.dims.w + slot] = i;
    }
    // KNOWN GAP (documented in the tire audit 2026-07-05): grains
    // beyond the per-cell capacity dims.w are DROPPED from the
    // neighbour list with no diagnostic — the consumer clamps to
    // min(count, dims.w). Overflow means silently lost contacts;
    // surfacing this needs a flag buffer + host check, tracked as a
    // follow-up rather than half-plumbed here.
}

@compute @workgroup_size(64)
fn step_dem_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.dims.x) {
        return;
    }

    let grain_mass = max(u.params0.x, 1.0e-9);
    let grain_radius = max(u.params0.y, 1.0e-9);
    let dt = max(u.params0.z, 0.0);
    let push_scale = max(u.params0.w, 0.0);
    let normal_stiffness = max(u.params1.x, 0.0);
    let damping = max(u.params1.y, 0.0);
    let friction = max(u.params1.z, 0.0);
    let floor_y = u.params1.w;

    let p0 = pos_in[i];
    let v0 = vel_in[i];
    var force = u.gravity.xyz * grain_mass;
    let c0 = cell_coords(p0.xyz);
    let min_c = vec3<u32>(max(vec3<i32>(c0) - vec3<i32>(1, 1, 1), vec3<i32>(0, 0, 0)));
    let max_shape = max(u.grid_shape, vec4<u32>(1u, 1u, 1u, 1u)).xyz - vec3<u32>(1u, 1u, 1u);
    let max_c = min(c0 + vec3<u32>(1u, 1u, 1u), max_shape);

    var z = min_c.z;
    loop {
        if (z > max_c.z) {
            break;
        }
        var y = min_c.y;
        loop {
            if (y > max_c.y) {
                break;
            }
            var x = min_c.x;
            loop {
                if (x > max_c.x) {
                    break;
                }
                let cell = cell_id(vec3<u32>(x, y, z));
                if (cell < u.dims.z) {
                    let range = cell_ranges[cell];
                    var cursor = range.x;
                    loop {
                        if (cursor >= range.y) {
                            break;
                        }
                        let j = sorted_indices[cursor];
                        if (j != i && j < u.dims.x) {
                            force += pair_force(
                                p0.xyz,
                                v0.xyz,
                                pos_in[j].xyz,
                                vel_in[j].xyz,
                                grain_radius * 2.0,
                                normal_stiffness,
                                damping,
                                friction
                            );
                        }
                        cursor += 1u;
                    }
                }
                x += 1u;
            }
            y += 1u;
        }
        z += 1u;
    }

    let pen = (floor_y + grain_radius) - p0.y;
    if (pen > 0.0) {
        let fn_mag = max(normal_stiffness * pen - damping * v0.y, 0.0);
        force.y += fn_mag;
        let vt = length(vec2<f32>(v0.x, v0.z));
        if (vt > 1.0e-9) {
            let ft = friction * fn_mag * (vt / (vt + 0.05));
            force.x -= v0.x * (ft / vt);
            force.z -= v0.z * (ft / vt);
        }
    }

    let delta = tire_delta(p0, grain_mass, grain_radius, dt, push_scale);
    var v = v0;
    v.x += force.x * dt / grain_mass + delta.x;
    v.y += force.y * dt / grain_mass + delta.y;
    v.z += force.z * dt / grain_mass + delta.z;

    var p = p0;
    p.x += v.x * dt;
    p.y += v.y * dt;
    p.z += v.z * dt;
    p.y -= delta.w;

    pos_out[i] = p;
    vel_out[i] = v;
}

@compute @workgroup_size(64)
fn step_dem_gpu_built_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.dims.x || u.dims.w == 0u) {
        return;
    }

    let grain_mass = max(u.params0.x, 1.0e-9);
    let grain_radius = max(u.params0.y, 1.0e-9);
    let dt = max(u.params0.z, 0.0);
    let push_scale = max(u.params0.w, 0.0);
    let normal_stiffness = max(u.params1.x, 0.0);
    let damping = max(u.params1.y, 0.0);
    let friction = max(u.params1.z, 0.0);
    let floor_y = u.params1.w;

    let p0 = pos_in[i];
    let v0 = vel_in[i];
    var force = u.gravity.xyz * grain_mass;
    let c0 = cell_coords(p0.xyz);
    let min_c = vec3<u32>(max(vec3<i32>(c0) - vec3<i32>(1, 1, 1), vec3<i32>(0, 0, 0)));
    let max_shape = max(u.grid_shape, vec4<u32>(1u, 1u, 1u, 1u)).xyz - vec3<u32>(1u, 1u, 1u);
    let max_c = min(c0 + vec3<u32>(1u, 1u, 1u), max_shape);

    var z = min_c.z;
    loop {
        if (z > max_c.z) {
            break;
        }
        var y = min_c.y;
        loop {
            if (y > max_c.y) {
                break;
            }
            var x = min_c.x;
            loop {
                if (x > max_c.x) {
                    break;
                }
                let cell = cell_id(vec3<u32>(x, y, z));
                if (cell < u.dims.z) {
                    let count = min(atomicLoad(&gpu_cell_counts[cell]), u.dims.w);
                    var cursor = 0u;
                    loop {
                        if (cursor >= count) {
                            break;
                        }
                        let j = gpu_cell_slots[cell * u.dims.w + cursor];
                        if (j != i && j < u.dims.x) {
                            force += pair_force(
                                p0.xyz,
                                v0.xyz,
                                pos_in[j].xyz,
                                vel_in[j].xyz,
                                grain_radius * 2.0,
                                normal_stiffness,
                                damping,
                                friction
                            );
                        }
                        cursor += 1u;
                    }
                }
                x += 1u;
            }
            y += 1u;
        }
        z += 1u;
    }

    let pen = (floor_y + grain_radius) - p0.y;
    if (pen > 0.0) {
        let fn_mag = max(normal_stiffness * pen - damping * v0.y, 0.0);
        force.y += fn_mag;
        let vt = length(vec2<f32>(v0.x, v0.z));
        if (vt > 1.0e-9) {
            let ft = friction * fn_mag * (vt / (vt + 0.05));
            force.x -= v0.x * (ft / vt);
            force.z -= v0.z * (ft / vt);
        }
    }

    let delta = tire_delta(p0, grain_mass, grain_radius, dt, push_scale);
    var v = v0;
    v.x += force.x * dt / grain_mass + delta.x;
    v.y += force.y * dt / grain_mass + delta.y;
    v.z += force.z * dt / grain_mass + delta.z;

    var p = p0;
    p.x += v.x * dt;
    p.y += v.y * dt;
    p.z += v.z * dt;
    p.y -= delta.w;

    pos_out[i] = p;
    vel_out[i] = v;
}

@compute @workgroup_size(64)
fn step_dem(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.dims.x) {
        return;
    }

    let grain_mass = max(u.params0.x, 1.0e-9);
    let grain_radius = max(u.params0.y, 1.0e-9);
    let dt = max(u.params0.z, 0.0);
    let push_scale = max(u.params0.w, 0.0);
    let normal_stiffness = max(u.params1.x, 0.0);
    let damping = max(u.params1.y, 0.0);
    let friction = max(u.params1.z, 0.0);
    let floor_y = u.params1.w;

    let p0 = pos_in[i];
    let v0 = vel_in[i];
    var force = u.gravity.xyz * grain_mass;

    var j = 0u;
    loop {
        if (j >= u.dims.x) {
            break;
        }
        if (j != i) {
            force += pair_force(
                p0.xyz,
                v0.xyz,
                pos_in[j].xyz,
                vel_in[j].xyz,
                grain_radius * 2.0,
                normal_stiffness,
                damping,
                friction
            );
        }
        j += 1u;
    }

    let pen = (floor_y + grain_radius) - p0.y;
    if (pen > 0.0) {
        let fn_mag = max(normal_stiffness * pen - damping * v0.y, 0.0);
        force.y += fn_mag;
        let vt = length(vec2<f32>(v0.x, v0.z));
        if (vt > 1.0e-9) {
            let ft = friction * fn_mag * (vt / (vt + 0.05));
            force.x -= v0.x * (ft / vt);
            force.z -= v0.z * (ft / vt);
        }
    }

    let delta = tire_delta(p0, grain_mass, grain_radius, dt, push_scale);
    var v = v0;
    v.x += force.x * dt / grain_mass + delta.x;
    v.y += force.y * dt / grain_mass + delta.y;
    v.z += force.z * dt / grain_mass + delta.z;

    var p = p0;
    p.x += v.x * dt;
    p.y += v.y * dt;
    p.z += v.z * dt;
    p.y -= delta.w;

    pos_out[i] = p;
    vel_out[i] = v;
}
