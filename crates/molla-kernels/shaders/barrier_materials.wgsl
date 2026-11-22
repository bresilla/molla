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
@group(1) @binding(4) var<storage, read> strain_neighbors: array<vec2<u32>>;
@group(1) @binding(5) var<storage, read> strain_springs: array<vec4<u32>>;
struct StrainTet { ids: vec4<u32>, data: vec4<f32> }
@group(1) @binding(6) var<storage, read> strain_tets: array<StrainTet>;
@group(2) @binding(0) var<storage, read> element_x: array<vec4<f32>>;
@group(2) @binding(1) var<storage, read> element_trial: array<vec4<f32>>;
@group(2) @binding(2) var<storage, read> element_direction: array<vec4<f32>>;
@group(2) @binding(3) var<storage, read> element_newton: array<vec4<f32>>;
@group(2) @binding(4) var<storage, read> element_previous: array<vec4<f32>>;
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

fn strain_cache_index(e: ElasticElement) -> u32 { return arrayLength(&elements)+e.config.z; }

fn strain_shape(e: ElasticElement, center: vec3<f32>, vertex: u32) -> vec3<f32> {
    for (var k = 0u; k < 3u; k++) {
        if (e.vertices[k] == vertex) { return element_previous[vertex].xyz-center; }
    }
    return vec3<f32>(0.0);
}

fn strain_reference(e: ElasticElement, center: vec3<f32>, neighbor: vec2<u32>) -> f32 {
    if (neighbor.x == 0u) {
        let other = elements[neighbor.y];
        var df: mat3x3<f32>;
        for (var k = 0u; k <= other.config.x; k++) {
            let q = strain_shape(e,center,other.vertices[k]);
            let w = element_shape(other,k);
            df += mat3x3<f32>(q*w.x,q*w.y,q*w.z);
        }
        var result = 0.0;
        for (var r = 0u; r < 9u; r++) {
            var product = 0.0;
            for (var c = 0u; c < 9u; c++) { product += element_cache[neighbor.y].hessian[9u*r+c]*df[c/3u][c%3u]; }
            result += df[r/3u][r%3u]*product;
        }
        return result;
    }
    if (neighbor.x == 1u) {
        let s = strain_springs[neighbor.y];
        let d = element_x[s.y].xyz-element_x[s.x].xyz;
        let len = max(length(d),1.0e-9); let n = d/len;
        let t = max(0.0,1.0-bitcast<f32>(s.z)/len);
        let q = strain_shape(e,center,s.y)-strain_shape(e,center,s.x);
        let axial = dot(n,q);
        return element_u.step.x*element_u.step.x*bitcast<f32>(s.w)*(t*dot(q,q)+(1.0-t)*axial*axial);
    }
    let tet = strain_tets[neighbor.y];
    let a = element_x[tet.ids.y].xyz-element_x[tet.ids.x].xyz;
    let b = element_x[tet.ids.z].xyz-element_x[tet.ids.x].xyz;
    let c = element_x[tet.ids.w].xyz-element_x[tet.ids.x].xyz;
    var g: array<vec3<f32>,4>;
    g[1] = cross(b,c)/6.0; g[2] = cross(c,a)/6.0; g[3] = cross(a,b)/6.0;
    g[0] = -g[1]-g[2]-g[3];
    var projected = 0.0;
    for (var k = 0u; k < 4u; k++) { projected += dot(g[k],strain_shape(e,center,tet.ids[k])); }
    return element_u.step.x*element_u.step.x*tet.data.x*projected*projected;
}

@compute @workgroup_size(64)
fn element_strain(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&elements)) { return; }
    let e = elements[i];
    if (!(e.material.w > 0.0)) { return; }
    let center = element_previous[e.vertices.x].xyz/3.0+element_previous[e.vertices.y].xyz/3.0+element_previous[e.vertices.z].xyz/3.0;
    let header = strain_neighbors[e.config.z];
    var stiffness = 0.0;
    var valid = element_cache[i].valid;
    for (var j = header.x; j < header.x+header.y; j++) {
        let neighbor = strain_neighbors[j];
        let term = strain_reference(e,center,neighbor);
        if (!material_finite(term)) { valid = 0u; }
        if (neighbor.x == 0u && element_cache[neighbor.y].valid == 0u) { valid = 0u; }
        stiffness += max(0.0,term);
    }
    let f = element_deformation(e,false);
    let gap = e.material.w-strain_largest_stretch(f);
    stiffness += element_u.step.x*element_u.step.x*e.rest1.w/(gap*gap);
    var response = strain_limit_evaluate(f,e.rest0.w,e.material.w);
    response.coefficient = stiffness;
    response.energy *= stiffness;
    response.gradient *= stiffness;
    for (var k = 0u; k < 81u; k++) {
        response.hessian[k] *= stiffness;
        if (!material_finite(response.hessian[k])) { valid = 0u; }
    }
    if (!(gap > 0.0) || !(stiffness > 0.0) || !material_finite(stiffness) || !material_finite(response.energy)
        || !element_vector_finite(response.gradient[0]) || !element_vector_finite(response.gradient[1])) { valid = 0u; }
    response.valid = min(response.valid,valid);
    element_cache[strain_cache_index(e)] = response;
    if (response.valid == 0u) { atomicOr(&element_scalars[13],32u); }
}

@compute @workgroup_size(64)
fn element_geometry(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&elements)) { return; }
    let e = elements[gid.x];
    let f = element_deformation(e,false);
    if (material_scale(f,e.config.x) == 0.0 || (e.material.w > 0.0 && !strain_inside(f,strain_safe_maximum(e.material.w)))) {
        atomicOr(&element_scalars[13],1u);
    }
}

@compute @workgroup_size(64)
fn element_prepare(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&elements)) { return; }
    let e = elements[i];
    let f = element_deformation(e,false);
    var r = material_evaluate(f,e.config.x,e.config.y,e.material.x,e.material.y,true);
    if (e.material.w > 0.0 && !strain_inside(f,strain_safe_maximum(e.material.w))) { r.valid = 0u; }
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
        let e = elements[a.x];
        var gradient = element_cache[a.x].gradient;
        if (e.material.w > 0.0) { gradient += element_cache[strain_cache_index(e)].gradient; }
        sum += gradient*element_shape(e,a.y);
    }
    element_gradient_out[i] = vec4<f32>(sum,0.0);
    if (!element_vector_finite(sum)) { atomicOr(&element_scalars[13],4u); }
}

fn element_stress(i: u32) -> mat3x3<f32> {
    let df = element_delta(elements[i],false);
    var stress: mat3x3<f32>;
    for (var r = 0u; r < 9u; r++) {
        for (var c = 0u; c < 9u; c++) {
            var value = element_cache[i].hessian[9u*r+c];
            if (elements[i].material.w > 0.0) { value += element_cache[strain_cache_index(elements[i])].hessian[9u*r+c]; }
            stress[r/3u][r%3u] += value*df[c/3u][c%3u];
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
    let f = element_deformation(e,true);
    let response = material_energy(f,e.config.x,e.config.y,e.material.x,e.material.y);
    var energy = response.energy*element_u.step.x*element_u.step.x*e.material.z;
    if (e.material.w > 0.0) {
        let barrier = strain_limit_energy(f,e.rest0.w,e.material.w);
        energy += element_cache[strain_cache_index(e)].coefficient*barrier.energy;
        if (barrier.valid == 0u) { energy = bitcast<f32>(0x7f800000u); }
    }
    if (response.valid == 0u || !material_finite(energy) || (e.material.w > 0.0 && !strain_inside(f,strain_safe_maximum(e.material.w)))) { energy = bitcast<f32>(0x7f800000u); }
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
    let e = elements[i];
    let df = element_delta(e,true);
    let norm = sqrt(dot(df[0],df[0])+dot(df[1],df[1])+dot(df[2],df[2]));
    var alpha = 1.0;
    if (norm > 0.0) { alpha = min(1.0,0.9*element_cache[i].minimum_stretch/norm); }
    if (e.material.w > 0.0) { alpha = strain_step_bound(element_deformation(e,false),df,e.material.w,alpha); }
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
