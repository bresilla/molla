@group(0) @binding(0) var<storage, read> status: u32;
@group(0) @binding(1) var<storage, read> source: array<u32>;
@group(0) @binding(2) var<storage, read_write> destination: array<u32>;

@compute @workgroup_size(64)
fn publish(@builtin(global_invocation_id) id: vec3<u32>) {
    if status != 0u || id.x >= arrayLength(&destination) {
        return;
    }
    destination[id.x] = source[id.x];
}
