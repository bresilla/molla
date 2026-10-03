@group(0) @binding(0) var<storage,read> cone_requests: array<ConeRequest>;
@group(0) @binding(1) var<storage,read_write> cone_results: array<ConeGeometry>;

@compute @workgroup_size(64)
fn evaluate_cones(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&cone_requests) { return; }
    cone_results[id.x] = cone_geometry(cone_requests[id.x]);
}
