@group(1) @binding(1) var<storage, read> reference_heights: array<f32>;

var<workgroup> surface_values: array<vec2<f32>, 64>;
var<workgroup> velocity_values: array<vec4<f32>, 64>;

fn bin_coords(x: f32, z: f32) -> vec2<u32> {
    let spacing = max(u.terrain.z, 1.0e-9);
    return vec2<u32>(
        u32(clamp(i32(floor((x - u.terrain.x) / spacing)), 0, i32(u.dims.y) - 1)),
        u32(clamp(i32(floor((z - u.terrain.y) / spacing)), 0, i32(u.dims.z) - 1))
    );
}

fn cell_index(p: vec4<f32>) -> u32 {
    let c = bin_coords(p.x, p.z);
    return c.y * u.dims.y + c.x;
}

fn cell_bounds(cell: u32) -> vec4<f32> {
    let spacing = max(u.terrain.z, 1.0e-9);
    let x0 = fma(f32(cell % u.dims.y), spacing, u.terrain.x);
    let z0 = fma(f32(cell / u.dims.y), spacing, u.terrain.y);
    return vec4<f32>(x0, z0, x0 + spacing, z0 + spacing);
}

fn contains(p: vec4<f32>, bounds: vec4<f32>) -> bool {
    return p.x >= bounds.x && p.x < bounds.z && p.z >= bounds.y && p.z < bounds.w;
}

@compute @workgroup_size(64)
fn reduce_cells(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let cell = group_index(wg);
    let valid = cell < u.dims.y * u.dims.z;
    let bounds = cell_bounds(min(cell, u.dims.y * u.dims.z - 1u));
    let first = bin_coords(bounds.x, bounds.y);
    let last = bin_coords(bounds.z, bounds.w);
    let radius = max(u.velocity.x, 1.0e-9);
    var height = u.terrain.w;
    var top = -3.4028234663852886e38;
    if valid && bounds.x < bounds.z && bounds.y < bounds.w {
        for (var z = first.y; z <= last.y; z += 1u) {
            for (var x = first.x; x <= last.x; x += 1u) {
                let bin = (z * u.dims.y + x) * 3u;
                let count = atomicLoad(&bins[bin]);
                let start = atomicLoad(&bins[bin + 1u]);
                for (var item = lane; item < count; item += 64u) {
                    let p = positions[members[start + item]];
                    if contains(p, bounds) {
                        height = max(height, p.y + radius);
                        top = max(top, p.y);
                    }
                }
            }
        }
    }
    surface_values[lane] = vec2<f32>(height, top);
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if lane < stride {
            surface_values[lane] = max(surface_values[lane], surface_values[lane + stride]);
        }
        workgroupBarrier();
    }
    let surface = surface_values[0];
    let band = max(radius * 2.25, max(u.terrain.z, 1.0e-9) * 0.25);
    var velocity = vec4<f32>(0.0);
    if valid && surface.y > -1.0e30 {
        for (var z = first.y; z <= last.y; z += 1u) {
            for (var x = first.x; x <= last.x; x += 1u) {
                let bin = (z * u.dims.y + x) * 3u;
                let count = atomicLoad(&bins[bin]);
                let start = atomicLoad(&bins[bin + 1u]);
                for (var item = lane; item < count; item += 64u) {
                    let id = members[start + item];
                    let p = positions[id];
                    if contains(p, bounds) && p.y >= surface.y - band {
                        velocity += vec4<f32>(velocities[id].xyz, 1.0);
                    }
                }
            }
        }
    }
    velocity_values[lane] = velocity;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if lane < stride {
            velocity_values[lane] += velocity_values[lane + stride];
        }
        workgroupBarrier();
    }
    if lane == 0u && valid {
        heights[cell] = surface.x;
        sinkage[cell] = max(reference_heights[cell] - surface.x, 0.0);
        let v = velocity_values[0];
        surface_velocity[cell] = vec4<f32>(v.xyz / max(v.w, 1.0), v.w);
    }
}

@compute @workgroup_size(64)
fn reference_cells(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let cell = group_index(wg) * 64u + lane;
    if cell >= u.dims.y * u.dims.z { return; }
    let bounds = cell_bounds(cell);
    let radius = max(u.velocity.x, 1.0e-9);
    var height = u.terrain.w;
    var top = -3.4028234663852886e38;
    for (var p = 0u; p < u.dims.x; p += 1u) {
        if contains(positions[p], bounds) {
            height = max(height, positions[p].y + radius);
            top = max(top, positions[p].y);
        }
    }
    heights[cell] = height;
    sinkage[cell] = max(reference_heights[cell] - height, 0.0);
    let band = max(radius * 2.25, max(u.terrain.z, 1.0e-9) * 0.25);
    var velocity = vec4<f32>(0.0);
    if top > -1.0e30 {
        for (var p = 0u; p < u.dims.x; p += 1u) {
            if contains(positions[p], bounds) && positions[p].y >= top - band {
                velocity += vec4<f32>(velocities[p].xyz, 1.0);
            }
        }
    }
    surface_velocity[cell] = vec4<f32>(velocity.xyz / max(velocity.w, 1.0), velocity.w);
}
