struct Uniforms {
    n: u32,
    shift: u32,
    num_blocks: u32,
    pad: u32,
}

@group(0) @binding(0) var<storage, read> keys_in: array<u32>;
@group(0) @binding(1) var<storage, read> values_in: array<u32>;
@group(0) @binding(2) var<storage, read_write> keys_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> values_out: array<u32>;
@group(0) @binding(4) var<storage, read_write> bins: array<u32>;
@group(1) @binding(0) var<uniform> u: Uniforms;

var<workgroup> scan_a: array<u32, 256>;
var<workgroup> scan_b: array<u32, 256>;
var<workgroup> carry: u32;

fn inclusive_scan(lane: u32, value: u32) -> u32 {
    scan_a[lane] = value;
    workgroupBarrier();
    for (var stride = 1u; stride < 256u; stride *= 2u) {
        var total = scan_a[lane];
        if lane >= stride { total += scan_a[lane - stride]; }
        scan_b[lane] = total;
        workgroupBarrier();
        scan_a[lane] = scan_b[lane];
        workgroupBarrier();
    }
    return scan_a[lane];
}

@compute @workgroup_size(256)
fn scan_block_bins(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let digit = wid.x;
    let lane = lid.x;
    if lane == 0u { carry = 0u; }
    workgroupBarrier();
    for (var start = 0u; start < u.num_blocks; start += 256u) {
        let block = start + lane;
        var value = 0u;
        if block < u.num_blocks { value = bins[digit * u.num_blocks + block]; }
        let prefix = inclusive_scan(lane, value);
        if block < u.num_blocks { bins[digit * u.num_blocks + block] = carry + prefix - value; }
        workgroupBarrier();
        if lane == 255u { carry += prefix; }
        workgroupBarrier();
    }
    if lane == 0u { bins[256u * u.num_blocks + digit] = carry; }
}

@compute @workgroup_size(256)
fn scan_digit_totals(@builtin(local_invocation_id) lid: vec3<u32>) {
    let index = 256u * u.num_blocks + lid.x;
    let value = bins[index];
    let prefix = inclusive_scan(lid.x, value);
    bins[index] = prefix - value;
}

var<workgroup> membership: array<atomic<u32>, 2048>;
var<workgroup> offsets: array<u32, 256>;

@compute @workgroup_size(256)
fn scatter_parallel(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let lane = lid.x;
    for (var i = lane; i < 2048u; i += 256u) { atomicStore(&membership[i], 0u); }
    offsets[lane] = bins[lane * u.num_blocks + wid.x] + bins[256u * u.num_blocks + lane];
    workgroupBarrier();
    let index = wid.x * 256u + lane;
    var key = 0u;
    var digit = 0u;
    if index < u.n {
        key = keys_in[index];
        digit = (key >> u.shift) & 255u;
        atomicOr(&membership[digit * 8u + lane / 32u], 1u << (lane % 32u));
    }
    workgroupBarrier();
    if index >= u.n { return; }
    var rank = 0u;
    for (var word = 0u; word < lane / 32u; word += 1u) {
        rank += countOneBits(atomicLoad(&membership[digit * 8u + word]));
    }
    let preceding = (1u << (lane % 32u)) - 1u;
    rank += countOneBits(atomicLoad(&membership[digit * 8u + lane / 32u]) & preceding);
    let destination = offsets[digit] + rank;
    keys_out[destination] = key;
    values_out[destination] = values_in[index];
}
