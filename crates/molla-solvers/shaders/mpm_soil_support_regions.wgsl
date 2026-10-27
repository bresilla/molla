struct SupportParams {
    origin: vec4<f32>,
    window: vec4<i32>,
    counts: vec4<u32>,
    geometry: vec4<f32>,
}
struct RegionBox { minimum: vec4<f32>, maximum: vec4<f32> }
@group(0) @binding(0) var<uniform> support: SupportParams;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> result: array<u32>;
@group(0) @binding(5) var<storage, read> regions: array<RegionBox>;

@compute @workgroup_size(64)
fn region_seeds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= support.counts.x { return; }
    let position = positions[pid].xyz;
    var requested = false;
    for (var r = 0u; r < support.counts.w; r++) {
        requested = requested || (all(position >= regions[r].minimum.xyz) && all(position <= regions[r].maximum.xyz));
    }
    result[pid] = select(0u, 1u, requested);
}
