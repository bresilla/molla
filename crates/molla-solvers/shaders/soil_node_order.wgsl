struct Node { arm_mass: vec4<f32>, momentum: vec4<f32> }
@group(0) @binding(0) var<storage, read> input_nodes: array<Node>;
@group(0) @binding(1) var<storage, read> spans: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> output_nodes: array<Node>;
@group(0) @binding(3) var<storage, read_write> status: atomic<u32>;
const CAPACITY: u32 = __CAPACITY__u;
var<workgroup> order: array<u32, __WIDTH__>;
var<workgroup> node_count: u32;
var<workgroup> fault: u32;

fn precedes(a: u32, b: u32, first: u32) -> bool {
    if a == b || a == 0xffffffffu { return false; }
    if b == 0xffffffffu { return true; }
    let x = input_nodes[first + a];
    let y = input_nodes[first + b];
    let xa = bitcast<vec4<u32>>(x.arm_mass);
    let ya = bitcast<vec4<u32>>(y.arm_mass);
    let xm = bitcast<vec4<u32>>(x.momentum);
    let ym = bitcast<vec4<u32>>(y.momentum);
    for (var k = 0u; k < 4u; k++) {
        if xa[k] != ya[k] { return xa[k] < ya[k]; }
    }
    for (var k = 0u; k < 4u; k++) {
        if xm[k] != ym[k] { return xm[k] < ym[k]; }
    }
    return false;
}

@compute @workgroup_size(64)
fn canonicalize(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    if lane == 0u {
        node_count = spans[group.x].y;
        fault = atomicLoad(&status);
    }
    let count = workgroupUniformLoad(&node_count);
    if workgroupUniformLoad(&fault) != 0u || count > CAPACITY { return; }
    let first = group.x * CAPACITY;
    var width = 1u;
    while width < count { width *= 2u; }
    for (var i = lane; i < width; i += 64u) {
        order[i] = select(0xffffffffu, i, i < count);
    }
    workgroupBarrier();
    for (var stage = 2u; stage <= width; stage *= 2u) {
        for (var stride = stage / 2u; stride > 0u; stride /= 2u) {
            for (var i = lane; i < width; i += 64u) {
                let j = i ^ stride;
                if j > i {
                    let a = order[i];
                    let b = order[j];
                    let ascending = (i & stage) == 0u;
                    if (ascending && precedes(b, a, first)) || (!ascending && precedes(a, b, first)) {
                        order[i] = b;
                        order[j] = a;
                    }
                }
            }
            workgroupBarrier();
        }
    }
    for (var i = lane; i < CAPACITY; i += 64u) {
        var node = Node(vec4<f32>(0.0), vec4<f32>(0.0));
        if i < count { node = input_nodes[first + order[i]]; }
        output_nodes[first + i] = node;
    }
}
