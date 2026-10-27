struct Params { leaves: u32, nodes: u32, blocks: u32, ratio: f32 }
struct Quality {
    baseline: f32, current: f32, checks: u32, rebuilds: u32,
    rebuild: u32, pad0: u32, pad1: u32, pad2: u32,
}
@group(0) @binding(0) var<storage, read> lower: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> upper: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> partial: array<f32>;
@group(0) @binding(3) var<storage, read_write> quality: Quality;
@group(0) @binding(4) var<storage, read_write> args: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;
var<workgroup> sums: array<f32, 256>;

fn reduce(value: f32, lane: u32) -> f32 {
    sums[lane] = value;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if lane < stride { sums[lane] += sums[lane + stride]; }
        workgroupBarrier();
    }
    return sums[0];
}

@compute @workgroup_size(256)
fn measure(@builtin(global_invocation_id) gid: vec3<u32>,
           @builtin(local_invocation_id) lid: vec3<u32>,
           @builtin(workgroup_id) wid: vec3<u32>) {
    let root = params.nodes - 1u;
    let extent = upper[root].xyz * 0.5 - lower[root].xyz * 0.5;
    let scale = max(max(extent.x, extent.y), extent.z);
    var area = 0.0;
    let i = params.leaves + gid.x;
    if i < params.nodes && scale > 0.0 {
        let r = extent / scale;
        let d = max(upper[i].xyz * 0.5 - lower[i].xyz * 0.5, vec3<f32>(0.0)) / scale;
        let root_area = dot(r, r.yzx);
        if root_area > 0.0 { area = dot(d, d.yzx) / root_area; }
    }
    let total = reduce(area, lid.x);
    if lid.x == 0u { partial[wid.x] = total; }
}

fn total_cost(lane: u32) -> f32 {
    var value = 0.0;
    for (var i = lane; i < params.blocks; i += 256u) { value += partial[i]; }
    return max(reduce(value, lane), 1.0);
}

@compute @workgroup_size(256)
fn decide(@builtin(local_invocation_id) lid: vec3<u32>) {
    let cost = total_cost(lid.x);
    if lid.x == 0u {
        let rebuild = quality.baseline == 0.0 || cost > quality.baseline * params.ratio;
        quality.current = cost;
        quality.checks += 1u;
        quality.rebuild = select(0u, 1u, rebuild);
        quality.rebuilds += quality.rebuild;
        args[0] = select(0u, (params.leaves + 63u) / 64u, rebuild);
        args[3] = select(0u, (params.leaves + 255u) / 256u, rebuild);
        args[6] = quality.rebuild;
        args[1] = 1u; args[2] = 1u;
        args[4] = 1u; args[5] = 1u;
        args[7] = 1u; args[8] = 1u;
    }
}

@compute @workgroup_size(256)
fn capture(@builtin(local_invocation_id) lid: vec3<u32>) {
    let cost = total_cost(lid.x);
    if lid.x == 0u {
        if quality.rebuild != 0u { quality.baseline = cost; }
        else { quality.baseline = min(quality.baseline, cost); }
        quality.current = cost;
    }
}
