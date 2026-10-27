struct Summary {
    maximum: atomic<u32>,
    active_count: atomic<u32>,
    padding0: u32,
    padding1: u32,
}
@group(0) @binding(0) var<storage, read> compaction: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> summary: Summary;
@group(0) @binding(2) var<uniform> params: vec4<u32>;
var<workgroup> maxima: array<u32, 64>;
var<workgroup> counts: array<u32, 64>;

@compute @workgroup_size(64)
fn summarize(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    var maximum = 0u;
    var active_count = 0u;
    if gid.x < params.x {
        let value = compaction[gid.x].x;
        if value > 0.0 { maximum = bitcast<u32>(value); }
        if value > bitcast<f32>(params.y) { active_count = 1u; }
    }
    maxima[lid.x] = maximum;
    counts[lid.x] = active_count;
    workgroupBarrier();
    var stride = 32u;
    loop {
        if lid.x < stride {
            maxima[lid.x] = max(maxima[lid.x], maxima[lid.x + stride]);
            counts[lid.x] += counts[lid.x + stride];
        }
        workgroupBarrier();
        if stride == 1u { break; }
        stride /= 2u;
    }
    if lid.x == 0u {
        atomicMax(&summary.maximum, maxima[0]);
        atomicAdd(&summary.active_count, counts[0]);
    }
}
