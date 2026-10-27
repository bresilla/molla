struct SupportParams {
    origin: vec4<f32>,
    window: vec4<i32>,
    counts: vec4<u32>,
    geometry: vec4<f32>,
}

@group(0) @binding(0) var<uniform> support: SupportParams;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> quiet: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> nodes: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> result: array<u32>;

fn support_base(pid: u32) -> vec3<i32> {
    let xp = (positions[pid].xyz - support.origin.xyz) * support.geometry.x;
    let base_xyz = vec3<f32>(floor(xp.x - 0.5), floor(xp.y - 0.5), floor(xp.z - 0.5));
    if !all(abs(base_xyz) <= vec3<f32>(536870912.0)) {
        return vec3<i32>(-1);
    }
    return vec3<i32>(i32(base_xyz.x), i32(base_xyz.y), i32(base_xyz.z)) - support.window.xyz;
}

fn support_valid(base: vec3<i32>) -> bool {
    return all(base >= vec3<i32>(0)) && all(base < vec3<i32>(i32(support.counts.y) - 2));
}

fn support_node(base: vec3<i32>, i: u32) -> u32 {
    let cell = vec3<u32>(base) + vec3<u32>(i % 3u, (i / 3u) % 3u, i / 9u);
    return cell.x + support.counts.y * (cell.y + support.counts.y * cell.z);
}

@compute @workgroup_size(64)
fn support_mark(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= support.counts.x { return; }
    let base = support_base(pid);
    if !support_valid(base) { result[pid] = 4u; return; }
    let kind = select(2u, 1u, quiet[pid].x >= support.counts.z);
    let requested = support.counts.w != 0u && result[pid] != 0u;
    result[pid] = kind | select(0u, 32u, requested);
    for (var i = 0u; i < 27u; i++) {
        atomicOr(&nodes[support_node(base, i)], kind | select(0u, 2u, requested));
    }
}

@compute @workgroup_size(64)
fn support_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= support.counts.x || result[pid] == 4u { return; }
    let base = support_base(pid);
    var neighbors = 0u;
    for (var i = 0u; i < 27u; i++) {
        neighbors |= atomicLoad(&nodes[support_node(base, i)]);
    }
    result[pid] |= neighbors << 3u;
}

fn closure_reached(base: vec3<i32>) -> bool {
    var neighbors = 0u;
    for (var i = 0u; i < 27u; i++) {
        neighbors |= atomicLoad(&nodes[support_node(base, i)]);
    }
    return (neighbors & 2u) != 0u;
}

@compute @workgroup_size(64)
fn closure_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= support.counts.x || result[pid] == 4u { return; }
    if closure_reached(support_base(pid)) { result[pid] |= 64u; }
}

@compute @workgroup_size(64)
fn closure_mark(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= support.counts.x || (result[pid] & 64u) == 0u { return; }
    let base = support_base(pid);
    for (var i = 0u; i < 27u; i++) {
        atomicOr(&nodes[support_node(base, i)], 2u);
    }
}

@compute @workgroup_size(64)
fn closure_check(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= support.counts.x || result[pid] == 4u { return; }
    if (result[pid] & 64u) == 0u && closure_reached(support_base(pid)) {
        result[pid] |= 128u;
    }
}
