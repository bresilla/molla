@group(1) @binding(1) var<storage, read> compaction: array<vec4<f32>>;
var<workgroup> compact_values: array<f32, 64>;
var<workgroup> velocity_values: array<vec3<f32>, 64>;

fn cell_index(p: vec4<f32>) -> u32 {
    let spacing = max(u.terrain.z, 1.0e-6);
    let x = u32(clamp(i32(floor((p.x - u.terrain.x) / spacing)), 0, i32(u.dims.y) - 1));
    let z = u32(clamp(i32(floor((p.z - u.terrain.y) / spacing)), 0, i32(u.dims.z) - 1));
    return z * u.dims.y + x;
}

@compute @workgroup_size(64)
fn reduce_cells(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let cell = group_index(wg);
    var count = 0u;
    var start = 0u;
    if cell < u.dims.y * u.dims.z {
        count = atomicLoad(&bins[cell * 3u]);
        start = atomicLoad(&bins[cell * 3u + 1u]);
    }
    var compact = 0.0;
    var velocity = vec3<f32>(0.0);
    for (var item = lane; item < count; item += 64u) {
        let p = members[start + item];
        compact = max(compact, compaction[p].x);
        velocity += velocities[p].xyz;
    }
    compact_values[lane] = compact;
    velocity_values[lane] = velocity;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if lane < stride {
            compact_values[lane] = max(compact_values[lane], compact_values[lane + stride]);
            velocity_values[lane] += velocity_values[lane + stride];
        }
        workgroupBarrier();
    }
    if lane == 0u && cell < u.dims.y * u.dims.z {
        sinkage[cell] = compact_values[0];
        heights[cell] = u.terrain.w - compact_values[0];
        var result = u.velocity.xyz;
        if count > 0u {
            result = velocity_values[0] / f32(count);
        }
        surface_velocity[cell] = vec4<f32>(result, 0.0);
    }
}

@compute @workgroup_size(64)
fn reference_cells(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let cell = group_index(wg) * 64u + lane;
    if cell >= u.dims.y * u.dims.z { return; }
    var compact = 0.0;
    var velocity = vec3<f32>(0.0);
    var count = 0u;
    for (var p = 0u; p < u.dims.x; p += 1u) {
        if cell_index(positions[p]) == cell {
            compact = max(compact, compaction[p].x);
            velocity += velocities[p].xyz;
            count += 1u;
        }
    }
    sinkage[cell] = compact;
    heights[cell] = u.terrain.w - compact;
    var result = u.velocity.xyz;
    if count > 0u { result = velocity / f32(count); }
    surface_velocity[cell] = vec4<f32>(result, 0.0);
}
