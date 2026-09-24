struct Params { source_rows: u32, capacity: u32, particles: u32, dofs: u32 }
@group(0) @binding(0) var<storage, read> material_diagonal: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> inverse_diagonal: array<f32>;
@group(0) @binding(2) var<uniform> u: Params;

@compute @workgroup_size(64)
fn pack(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < 3u*u.particles {
        inverse_diagonal[id.x] = material_diagonal[id.x/3u][id.x%3u];
    }
}
