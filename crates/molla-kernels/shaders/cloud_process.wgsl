struct Point {
    position: vec4<f32>,
    normal: vec4<f32>,
    direction: vec4<f32>,
    ids: vec4<u32>,
    grid: vec4<u32>,
    extra: vec4<u32>,
    angles: vec4<f32>,
};
struct Job {
    counts: vec4<u32>,
    grid: vec4<u32>,
    ids: vec4<u32>,
    translation: vec4<f32>,
    rotation: vec4<f32>,
    distance: vec4<f32>,
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    angles: vec4<f32>,
    misc: vec4<u32>,
};
struct Params { count: u32, jobs: u32, blocks: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> raw: array<u32>;
@group(0) @binding(1) var<storage, read_write> work: array<Point>;
@group(0) @binding(2) var<storage, read_write> output: array<Point>;
@group(0) @binding(3) var<storage, read_write> ranks: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> blocks: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read> jobs: array<Job>;
@group(0) @binding(6) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> params: Params;
var<workgroup> sums: array<u32, 256>;

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

fn finite(v: f32) -> bool { return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u; }
fn finite3(v: vec3<f32>) -> bool { return finite(v.x) && finite(v.y) && finite(v.z); }
fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}
fn read3(i: u32) -> vec3<f32> { return vec3<f32>(bitcast<f32>(raw[i]), bitcast<f32>(raw[i+1u]), bitcast<f32>(raw[i+2u])); }
fn axis(lo: f32, hi: f32, n: u32, i: u32) -> f32 {
    if n <= 1u { return lo; }
    return lo + (hi - lo) * (f32(i) / f32(n - 1u));
}
fn find_job(i: u32) -> u32 {
    var lo = 0u;
    var hi = params.jobs;
    while lo < hi {
        let mid = lo + (hi - lo) / 2u;
        if jobs[mid].counts.x + jobs[mid].counts.y <= i { lo = mid + 1u; }
        else { hi = mid; }
    }
    return lo;
}
fn scalar_pass(value: f32, threshold: f32, op: u32) -> bool {
    switch op {
        case 0u: { return value < threshold; }
        case 1u: { return value <= threshold; }
        case 2u: { return value > threshold; }
        case 3u: { return value >= threshold; }
        default: { return abs(value - threshold) < 1e-6; }
    }
}

@compute @workgroup_size(256)
fn prepare(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) local: u32,
           @builtin(workgroup_id) group: vec3<u32>) {
    let i = gid.x;
    var keep = 0u;
    if i < params.count {
        let j = find_job(i);
        let job = jobs[j];
        var point = work[i];
        if job.counts.z != 0u {
            let n = job.counts.y;
            let ray = i - job.counts.x;
            let base = job.counts.x * 14u;
            let shape = bitcast<i32>(raw[base + n * 13u + ray]);
            point.position = vec4<f32>(read3(base + ray * 4u), 1.0);
            point.normal = vec4<f32>(read3(base + n * 4u + ray * 4u), bitcast<f32>(raw[base + n * 12u + ray]));
            point.direction = vec4<f32>(read3(base + n * 8u + ray * 4u), 0.0);
            point.ids = job.ids;
            let pulse = ray / job.grid.z;
            point.grid = vec4<u32>(pulse / job.grid.x, pulse % job.grid.x, ray % job.grid.z, job.grid.w + ray);
            point.extra = vec4<u32>(bitcast<u32>(shape), 0xffffffffu, select(0u, 1u, shape >= 0), 0u);
            point.angles = vec4<f32>(0.0);
            if job.counts.z == 1u {
                point.angles = vec4<f32>(axis(job.angles.x, job.angles.y, job.grid.y, point.grid.x),
                    axis(job.angles.z, job.angles.w, job.grid.x, point.grid.y), 0.0, 0.0);
                point.extra.z |= 8u;
            }
        }
        var reason = 0u;
        if (point.extra.z & 1u) == 0u { reason = 2u; }
        else if !finite3(point.position.xyz) || !finite3(point.normal.xyz) || !finite3(point.direction.xyz)
            || !finite(point.position.w) || !finite(point.normal.w) || point.normal.w < 0.0
            || ((point.extra.z & 4u) != 0u && !finite(point.direction.w))
            || ((point.extra.z & 8u) != 0u && (!finite(point.angles.x) || !finite(point.angles.y))) { reason = 3u; }
        else {
            point.position = vec4<f32>(rotate(job.rotation, point.position.xyz) + job.translation.xyz, point.position.w);
            point.normal = vec4<f32>(rotate(job.rotation, point.normal.xyz), point.normal.w);
            point.direction = vec4<f32>(rotate(job.rotation, point.direction.xyz), point.direction.w);
            if !finite3(point.position.xyz) || !finite3(point.normal.xyz) || !finite3(point.direction.xyz) { reason = 3u; }
            else {
                let flags = job.counts.w;
                var accepted = true;
                if (flags & 1u) != 0u {
                    let delta = point.position.xyz - job.distance.xyz;
                    let largest = max(max(abs(delta.x), abs(delta.y)), abs(delta.z));
                    if largest > job.distance.w { accepted = false; }
                    else if largest > 0.0 { accepted = length(delta / largest) <= job.distance.w / largest; }
                }
                if (flags & 2u) != 0u {
                    let inside = all(point.position.xyz >= job.bounds_min.xyz) && all(point.position.xyz <= job.bounds_max.xyz);
                    accepted = accepted && (inside == ((flags & 4u) != 0u));
                }
                if (flags & 8u) != 0u { accepted = accepted && point.position.w >= job.translation.w; }
                if (flags & 16u) != 0u {
                    accepted = accepted && ((point.extra.z & 4u) != 0u) && scalar_pass(point.direction.w, job.bounds_min.w, job.misc.x);
                }
                if accepted { keep = 1u; work[i] = point; }
                else { reason = 4u; }
            }
        }
        if reason != 0u { atomicAdd(&counts[1u + j * 5u + reason], 1u); }
    }
    let prefix = scan(local, keep);
    if i < params.count { ranks[i] = vec2<u32>(prefix - keep, keep); }
    if local == 255u { blocks[group.x] = vec2<u32>(prefix, 0u); }
}

@compute @workgroup_size(256)
fn scan_blocks(@builtin(local_invocation_index) local: u32) {
    let chunk = (params.blocks + 255u) / 256u;
    let begin = min(local * chunk, params.blocks);
    let end = min(begin + chunk, params.blocks);
    var total = 0u;
    for (var i = begin; i < end; i++) { total += blocks[i].x; }
    let prefix = scan(local, total);
    var offset = prefix - total;
    for (var i = begin; i < end; i++) {
        blocks[i].y = offset;
        offset += blocks[i].x;
    }
    if local == 255u { atomicStore(&counts[0], prefix); }
}

fn global_prefix(i: u32) -> u32 {
    if i == params.count { return atomicLoad(&counts[0]); }
    return blocks[i / 256u].y + ranks[i].x;
}

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < params.count && ranks[i].y != 0u { output[global_prefix(i)] = work[i]; }
    if i < params.jobs {
        let job = jobs[i];
        let start = global_prefix(job.counts.x);
        let end = global_prefix(job.counts.x + job.counts.y);
        atomicStore(&counts[1u + i * 5u], start);
        atomicStore(&counts[2u + i * 5u], end - start);
    }
}
