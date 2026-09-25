@group(0) @binding(0) var<storage, read> source: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> destination: array<vec4<u32>>;

@compute @workgroup_size(64)
fn commit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let start = gid.x * 5u;
    if start >= arrayLength(&destination) { return; }
    for (var i = 0u; i < 5u; i++) { destination[start + i] = source[start + i]; }
}
