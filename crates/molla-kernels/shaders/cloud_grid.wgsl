struct Point {
    position: vec4<f32>, normal: vec4<f32>, direction: vec4<f32>,
    ids: vec4<u32>, grid: vec4<u32>, extra: vec4<u32>, angles: vec4<f32>,
};
struct Cell {
    header: vec4<u32>,
    mean: vec4<f32>,
    minimum: vec4<f32>,
    maximum: vec4<f32>,
};
struct Params {
    minimum: vec4<f32>, maximum: vec4<f32>, size: vec4<f32>,
    dimensions: vec4<u32>, counts: vec4<u32>,
};
@group(0) @binding(0) var<storage, read> points: array<Point>;
@group(0) @binding(1) var<storage, read_write> memberships: array<u32>;
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> scratch: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> blocks: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read_write> cells: array<Cell>;
@group(0) @binding(6) var<storage, read_write> diagnostics: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> params: Params;
var<workgroup> sums: array<u32, 256>;
var<workgroup> means: array<vec4<f32>, 256>;
var<workgroup> minima: array<vec4<f32>, 256>;
var<workgroup> maxima: array<vec4<f32>, 256>;

fn finite(v: f32) -> bool { return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u; }
fn finite4(v: vec4<f32>) -> bool { return finite(v.x) && finite(v.y) && finite(v.z) && finite(v.w); }
fn scan(local: u32, value: u32) -> u32 {
    sums[local] = value;
    workgroupBarrier();
    for (var offset = 1u; offset < 256u; offset *= 2u) {
        var previous = 0u;
        if local >= offset { previous = sums[local - offset]; }
        workgroupBarrier();
        sums[local] += previous;
        workgroupBarrier();
    }
    return sums[local];
}

@compute @workgroup_size(256)
fn classify(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.counts.x { return; }
    let p = points[i];
    memberships[i] = 0xffffffffu;
    indices[i] = 0xffffffffu;
    if (p.extra.z & 1u) == 0u { atomicAdd(&diagnostics[1], 1u); return; }
    if !finite4(p.position) { atomicAdd(&diagnostics[2], 1u); return; }
    if any(p.position.xyz < params.minimum.xyz) || any(p.position.xyz >= params.maximum.xyz) {
        atomicAdd(&diagnostics[3], 1u); return;
    }
    let grid = min(vec3<u32>(floor((p.position.xyz - params.minimum.xyz) / params.size.xyz)), params.dimensions.xyz - vec3<u32>(1u));
    let cell = (grid.z * params.dimensions.y + grid.y) * params.dimensions.x + grid.x;
    memberships[i] = cell;
    atomicAdd(&scratch[cell * 3u], 1u);
}

@compute @workgroup_size(256)
fn scan_cells(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) local: u32,
              @builtin(workgroup_id) group: vec3<u32>) {
    var count = 0u;
    if id.x < params.counts.y { count = atomicLoad(&scratch[id.x * 3u]); }
    let prefix = scan(local, count);
    if id.x < params.counts.y { atomicStore(&scratch[id.x * 3u + 1u], prefix - count); }
    if local == 255u { blocks[group.x] = vec2<u32>(prefix, 0u); }
}

@compute @workgroup_size(256)
fn scan_blocks(@builtin(local_invocation_index) local: u32) {
    let chunk = (params.counts.z + 255u) / 256u;
    let begin = min(local * chunk, params.counts.z);
    let end = min(begin + chunk, params.counts.z);
    var count = 0u;
    for (var i = begin; i < end; i++) { count += blocks[i].x; }
    let prefix = scan(local, count);
    var offset = prefix - count;
    for (var i = begin; i < end; i++) { blocks[i].y = offset; offset += blocks[i].x; }
    if local == 255u { atomicStore(&diagnostics[0], prefix); }
}

fn cell_start(cell: u32) -> u32 { return atomicLoad(&scratch[cell * 3u + 1u]) + blocks[cell / 256u].y; }

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.counts.x { return; }
    let cell = memberships[i];
    if cell == 0xffffffffu { return; }
    let offset = cell_start(cell) + atomicAdd(&scratch[cell * 3u + 2u], 1u);
    if offset >= params.counts.x { atomicAdd(&diagnostics[4], 1u); return; }
    indices[offset] = i;
}

@compute @workgroup_size(256)
fn reduce_cells(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) local: u32) {
    let cell = group.y * params.counts.w + group.x;
    if cell >= params.counts.y { return; }
    let count = atomicLoad(&scratch[cell * 3u]);
    let start = cell_start(cell);
    var mean = vec4<f32>(0.0);
    var minimum = vec4<f32>(3.402823466e38);
    var maximum = -minimum;
    for (var i = local; i < count; i += 256u) {
        let index = indices[start + i];
        if index >= params.counts.x { continue; }
        let value = points[index].position;
        mean += value / f32(count);
        minimum = min(minimum, value);
        maximum = max(maximum, value);
    }
    means[local] = mean;
    minima[local] = minimum;
    maxima[local] = maximum;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride /= 2u) {
        if local < stride {
            means[local] += means[local + stride];
            minima[local] = min(minima[local], minima[local + stride]);
            maxima[local] = max(maxima[local], maxima[local + stride]);
        }
        workgroupBarrier();
    }
    if local == 0u {
        var result = Cell(vec4<u32>(start, count, 0u, 0u), vec4<f32>(0.0), vec4<f32>(0.0), vec4<f32>(0.0));
        if count != 0u {
            result.minimum = minima[0];
            result.maximum = maxima[0];
            if finite4(means[0]) { result.mean = means[0]; }
            else { result.header.z = 1u; atomicAdd(&diagnostics[5], 1u); }
        }
        cells[cell] = result;
    }
}
