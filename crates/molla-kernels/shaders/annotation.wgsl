struct Label { shape: u32, class_id: u32 };
struct Box { bounds: vec4<u32>, identity: vec4<u32> };
@group(0) @binding(0) var<storage, read> shapes: array<u32>;
@group(0) @binding(1) var<storage, read_write> classes: array<u32>;
@group(0) @binding(2) var<storage, read> labels: array<Label>;
@group(0) @binding(3) var<storage, read> images: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> work: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> boxes: array<Box>;
@group(0) @binding(6) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> params: vec4<u32>;
var<workgroup> sums: array<u32, 256>;

@compute @workgroup_size(256)
fn initialize(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= params.w { return; }
    for (var j = 0u; j < 8u; j++) { atomicStore(&work[i * 8u + j], 0u); }
    atomicStore(&work[i * 8u], 0xffffffffu);
    atomicStore(&work[i * 8u + 1u], 0xffffffffu);
    boxes[i] = Box(vec4<u32>(0u), vec4<u32>(0u));
}

@compute @workgroup_size(256)
fn annotate(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixel = id.x;
    if pixel >= params.x { return; }
    var lo = 0u;
    var hi = params.y;
    while lo < hi {
        let mid = lo + (hi - lo) / 2u;
        if pixel >= images[mid].x + images[mid].y { lo = mid + 1u; }
        else { hi = mid; }
    }
    let camera = lo;
    let shape = shapes[pixel];
    classes[pixel] = 0u;
    if shape == 0xffffffffu { atomicAdd(&counts[camera * 4u + 2u], 1u); return; }
    lo = 0u;
    hi = params.z;
    while lo < hi {
        let mid = lo + (hi - lo) / 2u;
        if labels[mid].shape < shape { lo = mid + 1u; }
        else { hi = mid; }
    }
    if lo >= params.z { atomicAdd(&counts[camera * 4u + 1u], 1u); return; }
    if labels[lo].shape != shape { atomicAdd(&counts[camera * 4u + 1u], 1u); return; }
    classes[pixel] = labels[lo].class_id;
    atomicAdd(&counts[camera * 4u], 1u);
    let position = pixel - images[camera].x;
    let x = position % images[camera].z;
    let y = position / images[camera].z;
    let slot = (camera * params.z + lo) * 8u;
    atomicMin(&work[slot], x);
    atomicMin(&work[slot + 1u], y);
    atomicMax(&work[slot + 2u], x);
    atomicMax(&work[slot + 3u], y);
    atomicAdd(&work[slot + 6u], 1u);
}

@compute @workgroup_size(256)
fn compact(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) local: u32) {
    let camera = group.x;
    let chunk = (params.z + 255u) / 256u;
    let begin = min(local * chunk, params.z);
    let end = min(begin + chunk, params.z);
    var count = 0u;
    for (var i = begin; i < end; i++) {
        if atomicLoad(&work[(camera * params.z + i) * 8u + 6u]) != 0u { count++; }
    }
    sums[local] = count;
    workgroupBarrier();
    for (var offset = 1u; offset < 256u; offset *= 2u) {
        var previous = 0u;
        if local >= offset { previous = sums[local - offset]; }
        workgroupBarrier();
        sums[local] += previous;
        workgroupBarrier();
    }
    var destination = camera * params.z + sums[local] - count;
    for (var i = begin; i < end; i++) {
        let slot = (camera * params.z + i) * 8u;
        let pixels = atomicLoad(&work[slot + 6u]);
        if pixels == 0u { continue; }
        boxes[destination] = Box(vec4<u32>(atomicLoad(&work[slot]), atomicLoad(&work[slot + 1u]),
            atomicLoad(&work[slot + 2u]), atomicLoad(&work[slot + 3u])), vec4<u32>(labels[i].shape, labels[i].class_id, pixels, 0u));
        destination++;
    }
    if local == 255u { atomicStore(&counts[camera * 4u + 3u], sums[local]); }
}
