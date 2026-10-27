@group(0) @binding(0) var<storage, read> mappings: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> old_coordinates: array<f32>;
@group(0) @binding(2) var<storage, read> old_positions: array<f32>;
@group(0) @binding(3) var<storage, read_write> coordinates: array<f32>;

@compute @workgroup_size(64)
fn transfer(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&mappings) { return; }
    let mapping = mappings[id.x];
    var value = bitcast<f32>(mapping.z);
    if mapping.y == 1u { value += old_coordinates[mapping.x]; }
    if mapping.y == 2u { value += old_positions[mapping.x]; }
    coordinates[id.x] = value;
}
