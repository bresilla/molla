struct Row { point: vec4<f32>, normal: vec4<f32>, coefficients: vec4<f32>, endpoints: vec4<u32>,
    relative: vec4<f32>, surface: vec4<f32>, force: vec4<f32>, penalty: vec4<f32> }
struct Episode { key: vec2<u32>, onset: u32, pad: u32, normal: vec4<f32> }
struct Params { rows: u32, shapes: u32, capacity: u32, source_capacity: u32, speed: f32, similarity: f32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read_write> rows: array<Row>;
@group(0) @binding(1) var<storage, read> identities: array<u32>;
@group(0) @binding(2) var<storage, read_write> history: array<Episode>;
@group(0) @binding(3) var<storage, read_write> next: array<Episode>;
@group(0) @binding(4) var<storage, read_write> scratch: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> onsets: array<u32>;
@group(0) @binding(6) var<storage, read_write> impacts: array<Row>;
@group(0) @binding(7) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(8) var<uniform> u: Params;
const FAULT: u32 = 536870912u;
const NONE: u32 = 0xffffffffu;

fn finite(v: vec4<f32>) -> bool { return all(v == v) && all(abs(v) <= vec4<f32>(3.4028234e38)); }
fn hash(key: vec2<u32>) -> u32 {
    var value = key.x*0x9e3779b9u + key.y*0x85ebca6bu;
    value ^= value >> 16u; value *= 0x7feb352du; value ^= value >> 15u;
    return value & (u.capacity-1u);
}
fn key(endpoints: vec2<u32>) -> vec2<u32> {
    let a = identities[endpoints.x]; let b = identities[endpoints.y];
    return vec2<u32>(min(a,b),max(a,b));
}
fn find_current(k: vec2<u32>) -> u32 {
    let first = hash(k);
    for (var probe = 0u; probe < u.capacity; probe += 1u) {
        let slot = (first+probe) & (u.capacity-1u); let leader = atomicLoad(&scratch[slot]);
        if leader == 0u { return NONE; }
        if all(key(rows[leader-1u].endpoints.xy) == k) { return slot; }
    }
    return NONE;
}

@compute @workgroup_size(64)
fn clear(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < u.capacity {
        var empty: Episode; next[id.x] = empty;
        atomicStore(&scratch[id.x],0u); atomicStore(&scratch[u.capacity+id.x],0u);
    }
    if id.x < u.rows { onsets[id.x] = 0u; var empty: Row; impacts[id.x] = empty; }
}

@compute @workgroup_size(64)
fn group_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= u.rows || atomicLoad(&status) != 0u { return; }
    let row = rows[i];
    if row.coefficients.w == 0.0 { return; }
    if row.coefficients.w != 1.0 || any(row.endpoints.xy >= vec2<u32>(u.shapes))
        || !finite(row.normal) || !finite(row.relative) || !finite(row.point)
        || !finite(row.coefficients) || !finite(row.force) || !finite(row.surface)
        || row.normal.w < 0.0 || row.coefficients.y < 0.0 || row.coefficients.x < row.coefficients.y
        || row.coefficients.z < 0.0 || row.coefficients.z > 1.0
        || row.point.w < 0.0 || abs(dot(row.normal.xyz,row.normal.xyz)-1.0) > 1e-3 {
        atomicOr(&status,FAULT); return;
    }
    let k = key(row.endpoints.xy);
    if k.x == 0u || k.x == k.y { atomicOr(&status,FAULT); return; }
    let first = hash(k);
    var probe = 0u;
    for (var attempt = 0u; attempt < 2u*u.capacity; attempt += 1u) {
        let slot = (first+probe) & (u.capacity-1u);
        var leader = atomicLoad(&scratch[slot]);
        if leader == 0u {
            let claim = atomicCompareExchangeWeak(&scratch[slot],0u,i+1u);
            if claim.exchanged { leader = i+1u; }
            else { leader = claim.old_value; }
            if leader == 0u { continue; }
        }
        if all(key(rows[leader-1u].endpoints.xy) == k) {
            atomicMin(&scratch[slot],i+1u);
            let speed = dot(row.normal.xyz,row.relative.xyz);
            if speed != speed || abs(speed) > 3.4028234e38 { atomicOr(&status,FAULT); return; }
            if speed < -u.speed { atomicOr(&scratch[u.capacity+slot],1u); }
            return;
        }
        probe += 1u;
        if probe == u.capacity { break; }
    }
    atomicOr(&status,FAULT);
}

@compute @workgroup_size(64)
fn classify(@builtin(global_invocation_id) id: vec3<u32>) {
    let slot = id.x;
    if slot >= u.capacity || atomicLoad(&status) != 0u { return; }
    let leader = atomicLoad(&scratch[slot]);
    if leader == 0u { return; }
    let row = rows[leader-1u]; let k = key(row.endpoints.xy);
    let sign = select(-1.0,1.0,identities[row.endpoints.x] == k.x);
    let normal = normalize(row.normal.xyz)*sign;
    var retained = false;
    let first = hash(k);
    for (var probe = 0u; probe < u.capacity; probe += 1u) {
        let old = history[(first+probe) & (u.capacity-1u)];
        if old.key.x == 0u { break; }
        if all(old.key == k) { retained = dot(old.normal.xyz,normal) > u.similarity; break; }
    }
    let onset = !retained && atomicLoad(&scratch[u.capacity+slot]) != 0u;
    next[slot] = Episode(k,u32(onset),0u,vec4<f32>(normal,0.0));
}

@compute @workgroup_size(64)
fn publish_rows(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= u.rows || atomicLoad(&status) != 0u || rows[i].coefficients.w == 0.0 { return; }
    var row = rows[i]; let slot = find_current(key(row.endpoints.xy));
    if slot == NONE { atomicOr(&status,FAULT); return; }
    let onset = next[slot].onset;
    onsets[i] = onset;
    if onset != 0u {
        row.normal.w = 0.0; row.force = vec4<f32>(0.0);
        rows[i].normal.w = 0.0; rows[i].force = vec4<f32>(0.0);
    } else { row.coefficients.z = 0.0; }
    impacts[i] = row;
}

@compute @workgroup_size(64)
fn commit(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < u.capacity && atomicLoad(&status) == 0u { history[id.x] = next[id.x]; }
}

@compute @workgroup_size(64)
fn inherit(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if i >= u.source_capacity || atomicLoad(&status) != 0u { return; }
    let episode = history[i];
    if episode.key.x == 0u { return; }
    let first = hash(episode.key);
    var probe = 0u;
    for (var attempt = 0u; attempt < 2u*u.capacity; attempt += 1u) {
        let slot = (first+probe) & (u.capacity-1u);
        let claim = atomicCompareExchangeWeak(&scratch[slot],0u,i+1u);
        if claim.exchanged { next[slot] = episode; return; }
        if claim.old_value != 0u { probe += 1u; }
        if probe == u.capacity { break; }
    }
    atomicOr(&status,FAULT);
}
