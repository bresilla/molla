struct Row { point: vec4<f32>, normal: vec4<f32>, coefficients: vec4<f32>, endpoints: vec4<u32>,
    relative: vec4<f32>, surface: vec4<f32>, force: vec4<f32>, penalty: vec4<f32> }
struct ScalarRow { endpoints: vec4<u32>, velocity: vec4<f32> }
struct Params { arts: u32, bodies: u32, edges: u32, rows: u32, islands: u32, contacts: u32, scalars: u32, contact_limit: u32,
    impact: u32, onset_selection: u32, predicted_selection: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> rows: array<Row>;
@group(0) @binding(1) var<storage, read> owners: array<u32>;
@group(0) @binding(2) var<storage, read> edges: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(4) var<storage, read_write> parents: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> root_counts: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> keys: array<u32>;
@group(0) @binding(7) var<storage, read_write> values: array<u32>;
@group(0) @binding(8) var<storage, read_write> island_count: array<u32>;
@group(0) @binding(9) var<storage, read_write> counts: array<u32>;
@group(0) @binding(10) var<storage, read_write> indices: array<u32>;
@group(0) @binding(11) var<storage, read_write> row_slots: array<vec2<u32>>;
@group(0) @binding(12) var<storage, read_write> art_islands: array<u32>;
@group(0) @binding(13) var<storage, read_write> roots: array<u32>;
@group(0) @binding(14) var<uniform> u: Params;
@group(0) @binding(15) var<storage, read_write> art_offsets: array<u32>;
@group(0) @binding(16) var<storage, read_write> art_indices: array<u32>;
@group(0) @binding(17) var<storage, read> scalar_rows: array<ScalarRow>;
@group(0) @binding(18) var<storage, read> onsets: array<u32>;
@group(0) @binding(19) var<storage, read_write> selected: array<atomic<u32>>;
@group(0) @binding(20) var<storage, read> predicted_contacts: array<vec2<u32>>;
const NONE: u32 = 0xffffffffu;
const INPUT: u32 = 1048576u;
const CAPACITY: u32 = 2097152u;

fn finite(value: f32) -> bool { return value == value && abs(value) <= 3.4028234e38; }
fn excluded(art: u32) -> bool {
    if u.onset_selection != 2u || art == NONE { return false; }
    return onsets[art] != NONE;
}
fn validate_partition(a: u32, b: u32) {
    if a != NONE && b != NONE && excluded(a) != excluded(b) { atomicOr(&status, INPUT); }
}
fn selected_root(r: u32) -> bool {
    let flags = atomicLoad(&selected[r]);
    return (flags & 1u) != 0u && (u.predicted_selection != 1u || (flags & 2u) != 0u);
}
fn owner(body: u32) -> u32 {
    if (body == NONE) { return NONE; }
    if (body >= u.bodies) { atomicOr(&status, INPUT); return NONE; }
    return owners[body];
}
fn root(start: u32) -> u32 {
    var current = start;
    for (var step = 0u; step < u.arts + u.scalars; step += 1u) {
        let next = atomicLoad(&parents[current]);
        if (next == current) { return current; }
        current = next;
    }
    atomicOr(&status, INPUT);
    return NONE;
}
fn unite(first: u32, second: u32) {
    if (first == NONE || second == NONE) { return; }
    var a = first;
    var b = second;
    for (var step = 0u; step < 2u * (u.arts + u.scalars); step += 1u) {
        a = root(a); b = root(b);
        if (a == NONE || b == NONE || a == b) { return; }
        let lo = min(a, b); let hi = max(a, b);
        let old = atomicMin(&parents[hi], lo);
        if (old == hi) { return; }
        a = old; b = lo;
    }
    atomicOr(&status, INPUT);
}

@compute @workgroup_size(64)
fn clear(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (i == 0u) { island_count[0] = 0u; art_offsets[u.islands] = 0u; }
    if (i < max(u.arts + u.scalars, 1u)) { atomicStore(&parents[i], i); atomicStore(&root_counts[i], 0u); art_islands[i] = NONE; }
    if (i < max(u.arts + u.scalars, 1u)) { atomicStore(&selected[i], 0u); }
    if (i < max(u.arts + u.scalars, 1u)) { atomicStore(&root_counts[max(u.arts+u.scalars,1u)+i],0u); }
    if (i < max(u.arts, 1u)) { art_indices[i] = NONE; }
    if (i < u.rows + u.scalars) { keys[i] = NONE; values[i] = i; row_slots[i] = vec2<u32>(NONE); }
    if (i < u.islands) { counts[i] = 0u; roots[i] = NONE; art_offsets[i] = 0u; }
    if (i < u.islands * u.contacts) { indices[i] = NONE; }
}

@compute @workgroup_size(64)
fn join(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (atomicLoad(&status) != 0u) { return; }
    if (i < u.edges) { validate_partition(edges[i].x, edges[i].y); unite(edges[i].x, edges[i].y); }
    if (i < u.scalars) {
        let scalar = scalar_rows[i];
        if (scalar.endpoints.w > 1u) { atomicOr(&status, INPUT); return; }
        if (scalar.endpoints.w == 1u) {
            if (scalar.endpoints.z < 1u || scalar.endpoints.z > 3u) { atomicOr(&status, INPUT); return; }
            validate_partition(owner(scalar.endpoints.x), owner(scalar.endpoints.y));
            unite(u.arts + i, owner(scalar.endpoints.x));
            unite(u.arts + i, owner(scalar.endpoints.y));
        }
    }
    if (i >= u.rows) { return; }
    let row = rows[i];
    if (row.coefficients.w == 0.0) { return; }
    if (row.coefficients.w != 1.0 || !finite(row.normal.w) || row.normal.w < 0.0
        || !finite(row.coefficients.x) || !finite(row.coefficients.y)
        || row.coefficients.y < 0.0 || row.coefficients.x < row.coefficients.y) { atomicOr(&status, INPUT); return; }
    let a = owner(row.endpoints.z); let b = owner(row.endpoints.w);
    validate_partition(a, b);
    if (u.impact != 0u && a == NONE && b == NONE) { atomicOr(&status, INPUT); return; }
    if (u.impact != 0u || row.normal.w > 0.0) { unite(a, b); }
}

@compute @workgroup_size(64)
fn select_onsets(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (atomicLoad(&status) != 0u || i >= u.rows) { return; }
    let onset = onsets[i];
    var approach = 1u;
    if (u.predicted_selection != 0u) { approach = predicted_contacts[i].x; }
    if (onset > 1u || approach > 1u) { atomicOr(&status, INPUT); return; }
    if (rows[i].coefficients.w == 0.0) {
        if (onset != 0u || (u.predicted_selection != 0u && approach != 0u)) { atomicOr(&status, INPUT); }
        return;
    }
    if (onset == 0u && approach == 0u) { return; }
    let first = min(owner(rows[i].endpoints.z), owner(rows[i].endpoints.w));
    if (first == NONE) { atomicOr(&status, INPUT); return; }
    let r = root(first);
    if (r != NONE) { atomicOr(&selected[r], onset | (approach << 1u)); }
}

@compute @workgroup_size(64)
fn label(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (atomicLoad(&status) != 0u || i >= u.rows + u.scalars) { return; }
    if (i >= u.rows) {
        let index = i - u.rows;
        if (scalar_rows[index].endpoints.w == 0u) { return; }
        if excluded(owner(scalar_rows[index].endpoints.x)) || excluded(owner(scalar_rows[index].endpoints.y)) { return; }
        let r = root(u.arts + index);
        if (r == NONE) { return; }
        if (u.onset_selection == 1u && !selected_root(r)) { return; }
        keys[i] = r;
        atomicAdd(&root_counts[r], 1u);
        return;
    }
    let row = rows[i];
    if (row.coefficients.w == 0.0) { return; }
    let a = owner(row.endpoints.z); let b = owner(row.endpoints.w);
    if excluded(a) || excluded(b) { return; }
    let first = min(a, b);
    if (first == NONE) { return; }
    let r = root(first);
    if (r == NONE) { return; }
    if (u.impact == 0u && row.normal.w > 0.0) {
        atomicAdd(&root_counts[max(u.arts+u.scalars,1u)+r],1u);
    }
    if (u.impact == 0u && (row.normal.w == 0.0 || row.coefficients.x == 0.0)) { return; }
    if (u.predicted_selection != 0u && predicted_contacts[i].x == 0u) { return; }
    if (u.onset_selection == 1u && !selected_root(r)) { return; }
    if (u.impact != 0u) {
        atomicAdd(&root_counts[max(u.arts+u.scalars,1u)+r],1u);
    }
    keys[i] = r;
    atomicAdd(&root_counts[r], 1u);
}

@compute @workgroup_size(64)
fn compact(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x != 0u || atomicLoad(&status) != 0u) { return; }
    var total = 0u;
    for (var a = 0u; a < u.arts + u.scalars; a += 1u) {
        let count = atomicLoad(&root_counts[a]);
        if (count == 0u) { continue; }
        if (total < u.islands) { roots[total] = a; counts[total] = count; art_islands[a] = total; }
        if (count > u.contacts || total >= u.islands
            || atomicLoad(&root_counts[max(u.arts+u.scalars,1u)+a]) > u.contact_limit) { atomicOr(&status, CAPACITY); }
        total += 1u;
    }
    island_count[0] = total;
    for (var a = 0u; a < u.arts; a += 1u) {
        let r = root(a);
        if (r != NONE) { art_islands[a] = art_islands[r]; }
    }
    if (atomicLoad(&status) != 0u) { return; }
    for (var i = 0u; i < total; i += 1u) { atomicStore(&root_counts[roots[i]], 0u); }
    for (var a = 0u; a < u.arts; a += 1u) {
        let island = art_islands[a];
        if (island != NONE) { atomicAdd(&root_counts[roots[island]], 1u); }
    }
    var offset = 0u;
    for (var i = 0u; i < u.islands; i += 1u) {
        art_offsets[i] = offset;
        if (i < total) { offset += atomicExchange(&root_counts[roots[i]], 0u); }
    }
    art_offsets[u.islands] = offset;
    for (var a = 0u; a < u.arts; a += 1u) {
        let island = art_islands[a];
        if (island != NONE) { art_indices[art_offsets[island] + atomicAdd(&root_counts[roots[island]], 1u)] = a; }
    }
}

@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    if (atomicLoad(&status) != 0u || i >= u.rows + u.scalars) { return; }
    let key = keys[i];
    if (key == NONE) { return; }
    var lo = 0u; var hi = i;
    while (lo < hi) {
        let mid = lo + (hi - lo) / 2u;
        if (keys[mid] < key) { lo = mid + 1u; } else { hi = mid; }
    }
    let slot = i - lo;
    let island = art_islands[key];
    if (slot >= u.contacts || island >= u.islands) { atomicOr(&status, CAPACITY); return; }
    let original = values[i];
    indices[island * u.contacts + slot] = original;
    row_slots[original] = vec2<u32>(island, slot);
}
