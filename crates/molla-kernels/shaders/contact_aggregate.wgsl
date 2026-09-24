// One invocation per body reduces its contact CSR into count, maximum depth,
// unweighted signed normals and mean world-space contact point.

const B_SIDE_BIT: u32 = 0x80000000u;

struct Uniforms {
    body_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> contact_a:                array<i32>;
@group(0) @binding(1) var<storage, read> contact_b:                array<i32>;
@group(0) @binding(2) var<storage, read> contact_normal:           array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> contact_depth:            array<f32>;
@group(0) @binding(4) var<storage, read> contact_point:            array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> body_contact_offsets:     array<u32>;
@group(0) @binding(6) var<storage, read> body_contact_indices:     array<u32>;

@group(1) @binding(0) var<storage, read_write> out_count:    array<u32>;
@group(1) @binding(1) var<storage, read_write> out_max_depth: array<f32>;
@group(1) @binding(2) var<storage, read_write> out_normal_sum: array<vec4<f32>>;
@group(1) @binding(3) var<storage, read_write> out_centroid:   array<vec4<f32>>;

@group(2) @binding(0) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64)
fn aggregate_contacts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.body_count) { return; }

    let begin = body_contact_offsets[tid];
    let end   = body_contact_offsets[tid + 1u];
    let count = end - begin;

    var max_d: f32 = 0.0;
    var n_sum: vec3<f32> = vec3<f32>(0.0);
    var p_sum: vec3<f32> = vec3<f32>(0.0);

    for (var i: u32 = begin; i < end; i = i + 1u) {
        let packed = body_contact_indices[i];
        let c = packed & ~B_SIDE_BIT;
        let is_b = (packed & B_SIDE_BIT) != 0u;

        let n = contact_normal[c].xyz;
        let d = contact_depth[c];
        let p = contact_point[c].xyz;

        max_d = max(max_d, d);
        // Normals point a→b: negative on the a-side, positive on the b-side.
        let signed_n = select(-n, n, is_b);
        n_sum = n_sum + signed_n;
        p_sum = p_sum + p;
    }

    out_count[tid] = count;
    out_max_depth[tid] = max_d;
    out_normal_sum[tid] = vec4<f32>(n_sum, 0.0);
    if (count > 0u) {
        out_centroid[tid] = vec4<f32>(p_sum / f32(count), 0.0);
    } else {
        out_centroid[tid] = vec4<f32>(0.0);
    }
}
