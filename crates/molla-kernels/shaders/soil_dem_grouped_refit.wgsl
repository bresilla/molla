struct Params {
    dims: vec4<u32>, params0: vec4<f32>, params1: vec4<f32>, gravity: vec4<f32>,
    grid_origin_cell_size: vec4<f32>, grid_shape: vec4<u32>,
}
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> tree: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> indices: array<u32>;
@group(1) @binding(0) var<uniform> u: Params;
var<workgroup> local_min: array<vec3<f32>, 127>;
var<workgroup> local_max: array<vec3<f32>, 127>;

fn write_node(node: u32, lo: vec3<f32>, hi: vec3<f32>, left: f32, right: f32, payload: vec4<f32>) {
    let offset = 12u * node;
    for (var a = 0u; a < 3u; a++) {
        atomicStore(&tree[offset+a], bitcast<u32>(lo[a]));
        atomicStore(&tree[offset+4u+a], bitcast<u32>(hi[a]));
    }
    atomicStore(&tree[offset+3u], bitcast<u32>(left));
    atomicStore(&tree[offset+7u], bitcast<u32>(right));
    for (var a = 0u; a < 4u; a++) { atomicStore(&tree[offset+8u+a], bitcast<u32>(payload[a])); }
}

fn expand_bound(offset: u32, value: f32, lower: bool) {
    var old = atomicLoad(&tree[offset]);
    loop {
        let previous = bitcast<f32>(old);
        let next = bitcast<u32>(select(max(previous, value), min(previous, value), lower));
        if next == old { return; }
        let result = atomicCompareExchangeWeak(&tree[offset], old, next);
        if result.exchanged { return; }
        old = result.old_value;
    }
}

@compute @workgroup_size(64)
fn reset_top(@builtin(global_invocation_id) id: vec3<u32>) {
    let groups = u.dims.y / min(u.dims.y, 64u);
    if id.x >= groups-1u { return; }
    write_node(id.x, vec3<f32>(3.4028234663852886e38), vec3<f32>(-3.4028234663852886e38),
        f32(2u*id.x+1u), f32(2u*id.x+2u), vec4<f32>(0.0));
}

@compute @workgroup_size(64)
fn refit_groups(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    let batch = min(u.dims.y, 64u);
    let leaf = group.x * batch + lane;
    if lane < batch {
        var lo = vec3<f32>(3.4028234663852886e38);
        var hi = vec3<f32>(-3.4028234663852886e38);
        var start = u.dims.x;
        var count = 0u;
        if leaf < u.grid_shape.w {
            start = leaf * max(u.dims.z, 1u);
            let end = min(start + max(u.dims.z, 1u), u.dims.x);
            count = end-start;
            var first = indices[start];
            if first >= u.dims.x { first = 0u; }
            lo = positions[first].xyz; hi = lo;
            for (var cursor = start+1u; cursor < end; cursor++) {
                let particle = indices[cursor];
                if particle < u.dims.x {
                    lo = min(lo, positions[particle].xyz);
                    hi = max(hi, positions[particle].xyz);
                }
            }
        }
        local_min[batch-1u+lane] = lo;
        local_max[batch-1u+lane] = hi;
        write_node(u.grid_shape.x+leaf, lo, hi, f32(0xffffffffu), f32(0xffffffffu),
            vec4<f32>(f32(start), f32(count), 1.0, 0.0));
    }
    workgroupBarrier();
    var width = batch/2u;
    var base = u.dims.y/2u;
    loop {
        if width == 0u { break; }
        if lane < width {
            let node = width-1u+lane;
            let lo = min(local_min[2u*node+1u], local_min[2u*node+2u]);
            let hi = max(local_max[2u*node+1u], local_max[2u*node+2u]);
            local_min[node] = lo; local_max[node] = hi;
            let global = base-1u+group.x*width+lane;
            write_node(global, lo, hi, f32(2u*global+1u), f32(2u*global+2u), vec4<f32>(0.0));
        }
        workgroupBarrier();
        width /= 2u;
        base /= 2u;
    }
    if lane == 0u {
        var node = u.dims.y/batch-1u+group.x;
        loop {
            if node == 0u { break; }
            node = (node-1u)/2u;
            for (var a = 0u; a < 3u; a++) {
                expand_bound(12u*node+a, local_min[0][a], true);
                expand_bound(12u*node+4u+a, local_max[0][a], false);
            }
        }
    }
}
