struct ElementUniforms { counts: vec4<u32>, step: vec4<f32> }
struct ElasticElement {
    vertices: vec4<u32>,
    rest0: vec4<f32>,
    rest1: vec4<f32>,
    rest2: vec4<f32>,
    material: vec4<f32>,
    config: vec4<u32>,
}

@group(0) @binding(0) var<uniform> element_u: ElementUniforms;
@group(1) @binding(0) var<storage, read> elements: array<ElasticElement>;
@group(1) @binding(1) var<storage, read> element_offsets: array<u32>;
@group(1) @binding(2) var<storage, read> element_adjacency: array<vec2<u32>>;
@group(1) @binding(3) var<storage, read> element_pinned: array<u32>;
@group(2) @binding(0) var<storage, read> element_x: array<vec4<f32>>;
@group(2) @binding(1) var<storage, read> element_trial: array<vec4<f32>>;
@group(2) @binding(2) var<storage, read> element_direction: array<vec4<f32>>;
@group(2) @binding(3) var<storage, read> element_newton: array<vec4<f32>>;
@group(3) @binding(0) var<storage, read_write> element_gradient_out: array<vec4<f32>>;
@group(3) @binding(1) var<storage, read_write> element_hessian_out: array<vec4<f32>>;
@group(3) @binding(2) var<storage, read_write> element_partial: array<f32>;
@group(3) @binding(3) var<storage, read_write> element_scalars: array<atomic<u32>>;
@group(3) @binding(4) var<storage, read_write> element_cache: array<MaterialResponse>;
@group(3) @binding(5) var<storage, read_write> element_scratch: array<mat3x3<f32>>;

fn element_rest(e: ElasticElement) -> mat3x3<f32> {
    return mat3x3<f32>(e.rest0.xyz, e.rest1.xyz, e.rest2.xyz);
}

fn element_shape(e: ElasticElement, local: u32) -> vec3<f32> {
    let rows = transpose(element_rest(e));
    if (local == 0u) { return -rows[0]-rows[1]-rows[2]; }
    return rows[local-1u];
}

fn element_deformation(e: ElasticElement, trial: bool) -> mat3x3<f32> {
    var points: array<vec3<f32>,4>;
    for (var k = 0u; k <= e.config.x; k++) {
        points[k] = element_x[e.vertices[k]].xyz;
        if (trial) { points[k] = element_trial[e.vertices[k]].xyz; }
    }
    var c = vec3<f32>(0.0);
    if (e.config.x == 3u) { c = points[3]-points[0]; }
    return mat3x3<f32>(points[1]-points[0],points[2]-points[0],c)*element_rest(e);
}

fn element_delta(e: ElasticElement, newton: bool) -> mat3x3<f32> {
    var result: mat3x3<f32>;
    for (var k = 0u; k <= e.config.x; k++) {
        let id = e.vertices[k];
        if (element_pinned[id] != 0u) { continue; }
        var d = element_direction[id].xyz;
        if (newton) { d = element_newton[id].xyz; }
        let shape = element_shape(e,k);
        result += mat3x3<f32>(d*shape.x,d*shape.y,d*shape.z);
    }
    return result;
}

fn element_vector_finite(v: vec3<f32>) -> bool {
    return material_finite(v.x) && material_finite(v.y) && material_finite(v.z);
}

@compute @workgroup_size(64)
fn element_geometry(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&elements)) { return; }
    let e = elements[gid.x];
    if (material_scale(element_deformation(e,false),e.config.x) == 0.0) {
        atomicOr(&element_scalars[13],1u);
    }
}

@compute @workgroup_size(64)
fn element_prepare(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&elements)) { return; }
    let e = elements[i];
    var r = material_evaluate(element_deformation(e,false),e.config.x,e.config.y,e.material.x,e.material.y,true);
    let factor = element_u.step.x*element_u.step.x*e.material.z;
    if (!material_finite(factor) || !(factor > 0.0)) { r.valid = 0u; }
    r.energy *= factor;
    r.gradient *= factor;
    for (var k = 0u; k < 81u; k++) {
        r.hessian[k] *= factor;
        if (!material_finite(r.hessian[k])) { r.valid = 0u; }
    }
    if (!material_finite(r.energy) || !element_vector_finite(r.gradient[0]) || !element_vector_finite(r.gradient[1]) || !element_vector_finite(r.gradient[2])) { r.valid = 0u; }
    element_cache[i] = r;
    if (r.valid == 0u) { atomicOr(&element_scalars[13],2u); }
}

@compute @workgroup_size(64)
fn element_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= element_u.counts.x || element_pinned[i] != 0u) { return; }
    var sum = element_gradient_out[i].xyz;
    for (var j = element_offsets[i]; j < element_offsets[i+1u]; j++) {
        let a = element_adjacency[j];
        sum += element_cache[a.x].gradient*element_shape(elements[a.x],a.y);
    }
    element_gradient_out[i] = vec4<f32>(sum,0.0);
    if (!element_vector_finite(sum)) { atomicOr(&element_scalars[13],4u); }
}

fn element_stress(i: u32) -> mat3x3<f32> {
    let df = element_delta(elements[i],false);
    var stress: mat3x3<f32>;
    for (var r = 0u; r < 9u; r++) {
        for (var c = 0u; c < 9u; c++) {
            stress[r/3u][r%3u] += element_cache[i].hessian[9u*r+c]*df[c/3u][c%3u];
        }
    }
    return stress;
}

@compute @workgroup_size(64)
fn element_product(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&elements)) { return; }
    element_scratch[i] = element_stress(i);
}

@compute @workgroup_size(64)
fn element_hessian(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= element_u.counts.x || element_pinned[i] != 0u) { return; }
    var sum = element_hessian_out[i].xyz;
    for (var j = element_offsets[i]; j < element_offsets[i+1u]; j++) {
        let a = element_adjacency[j];
        sum += element_scratch[a.x]*element_shape(elements[a.x],a.y);
    }
    element_hessian_out[i] = vec4<f32>(sum,0.0);
    if (!element_vector_finite(sum)) { atomicOr(&element_scalars[13],8u); }
}

@compute @workgroup_size(64)
fn element_energy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&elements)) { return; }
    let e = elements[i];
    let response = material_energy(element_deformation(e,true),e.config.x,e.config.y,e.material.x,e.material.y);
    var energy = response.energy*element_u.step.x*element_u.step.x*e.material.z;
    if (response.valid == 0u || !material_finite(energy)) { energy = bitcast<f32>(0x7f800000u); }
    element_scratch[i][0][0] = energy;
}

@compute @workgroup_size(64)
fn element_sum_energy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= element_u.counts.x) { return; }
    var energy = element_partial[i];
    for (var j = element_offsets[i]; j < element_offsets[i+1u]; j++) {
        let a = element_adjacency[j];
        if (a.y == 0u) { energy += element_scratch[a.x][0][0]; }
    }
    element_partial[i] = energy;
}

@compute @workgroup_size(64)
fn element_bound(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&elements)) { return; }
    let df = element_delta(elements[i],true);
    let norm = sqrt(dot(df[0],df[0])+dot(df[1],df[1])+dot(df[2],df[2]));
    var alpha = 1.0;
    if (norm > 0.0) { alpha = min(1.0,0.9*element_cache[i].minimum_stretch/norm); }
    if (!material_finite(norm) || !material_finite(alpha) || element_cache[i].valid == 0u) {
        alpha = 0.0;
        atomicOr(&element_scalars[13],16u);
    }
    element_scratch[i][0][0] = alpha;
}

var<workgroup> element_reduction: array<f32,64>;

@compute @workgroup_size(64)
fn element_reduce_bound(@builtin(local_invocation_id) lid: vec3<u32>) {
    var alpha = 1.0;
    for (var i = lid.x; i < arrayLength(&elements); i += 64u) {
        alpha = min(alpha,element_scratch[i][0][0]);
    }
    element_reduction[lid.x] = alpha;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride /= 2u) {
        if (lid.x < stride) { element_reduction[lid.x] = min(element_reduction[lid.x],element_reduction[lid.x+stride]); }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        atomicStore(&element_scalars[6],bitcast<u32>(min(bitcast<f32>(atomicLoad(&element_scalars[6])),element_reduction[0])));
    }
}
