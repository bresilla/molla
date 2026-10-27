struct Params { tets: u32, particles: u32, pad0: u32, pad1: u32 }
struct Diagnostics { minimum: atomic<u32>, bad_tet: atomic<u32> }
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> reference: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> inverse_rest: array<mat3x3<f32>>;
@group(0) @binding(4) var<storage, read_write> diagnostics: Diagnostics;
@group(0) @binding(5) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(6) var<uniform> params: Params;

fn fail(tet: u32, flag: u32) {
    atomicOr(&status, flag);
    atomicMin(&diagnostics.minimum, 0u);
    atomicMin(&diagnostics.bad_tet, tet);
}

@compute @workgroup_size(64)
fn measure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tet = gid.x;
    if tet >= params.tets { return; }
    var nodes: array<u32, 4>;
    for (var c = 0u; c < 4u; c += 1u) {
        let node = indices[4u*tet+c];
        if node >= params.particles { fail(tet, 8u); return; }
        nodes[c] = node;
    }
    var edges: array<vec3<f32>, 3>;
    for (var c = 0u; c < 3u; c += 1u) {
        edges[c] = (positions[nodes[c+1u]].xyz - positions[nodes[0]].xyz)
            - (reference[nodes[c+1u]].xyz - reference[nodes[0]].xyz);
    }
    let du = mat3x3<f32>(edges[0], edges[1], edges[2]);
    let identity = mat3x3<f32>(vec3<f32>(1.0,0.0,0.0), vec3<f32>(0.0,1.0,0.0), vec3<f32>(0.0,0.0,1.0));
    let j = determinant(identity + du * inverse_rest[tet]);
    if !(abs(j) <= 3.402823e38) { fail(tet, 4u); return; }
    if j <= 0.0 { fail(tet, 8u); return; }
    atomicMin(&diagnostics.minimum, bitcast<u32>(j));
}
