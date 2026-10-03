
@group(3) @binding(1) var<storage, read> resident_count: array<u32>;
@group(3) @binding(2) var<storage, read_write> resident_status: array<atomic<u32>>;
@group(3) @binding(3) var<storage, read> collision_permissions: array<vec4<u32>>;
@group(3) @binding(4) var<storage, read> collision_exclusions: array<vec2<u32>>;
@group(0) @binding(5) var<storage, read> plane_position_residues: array<vec4<f32>>;

const FAULT_CAPACITY: u32 = 1u;
const FAULT_INDEX: u32 = 2u;
const FAULT_GJK: u32 = 4u;
const FAULT_EPA: u32 = 8u;
const FAULT_NONFINITE: u32 = 16u;

fn finite4(v: vec4<f32>) -> bool {
    return all(v == v) && all(abs(v) <= vec4<f32>(3.4028234e38));
}

fn resident_shape_valid(shape: u32) -> bool {
    let body = shape_body[shape];
    if (body < -1 || (body >= 0 && u32(body) >= arrayLength(&body_q))) {
        atomicOr(&resident_status[0], FAULT_INDEX);
        return false;
    }
    let local = shape_transform[shape];
    if (!finite4(shape_geom_data[shape]) || !finite4(local.position) || !finite4(local.rotation)) {
        atomicOr(&resident_status[0], FAULT_NONFINITE);
        return false;
    }
    if (body >= 0) {
        let pose = body_q[u32(body)];
        if (!finite4(pose.position) || !finite4(pose.rotation) || !finite4(plane_position_residues[u32(body)])) {
            atomicOr(&resident_status[0], FAULT_NONFINITE);
            return false;
        }
    }
    let world = world_xf_of(shape);
    if (!finite4(world.position) || !finite4(world.rotation)) {
        atomicOr(&resident_status[0], FAULT_NONFINITE);
        return false;
    }
    return true;
}

fn resident_pair_valid(i: u32) -> bool {
    let a = pair_a[i];
    let b = pair_b[i];
    if (a >= u.shape_count || b >= u.shape_count || a == b) {
        atomicOr(&resident_status[0], FAULT_INDEX);
        return false;
    }
    let ba = shape_body[a];
    let bb = shape_body[b];
    if (ba < -1 || bb < -1 || (ba >= 0 && u32(ba) >= arrayLength(&body_q))
        || (bb >= 0 && u32(bb) >= arrayLength(&body_q))) {
        atomicOr(&resident_status[0], FAULT_INDEX);
        return false;
    }
    return true;
}

fn resident_pair_allowed(i: u32) -> bool {
    let a = pair_a[i];
    let b = pair_b[i];
    let pa = collision_permissions[a];
    let pb = collision_permissions[b];
    if ((pa.x & 1u) == 0u || (pb.x & 1u) == 0u) { return false; }
    if (u._pad1 == 0u && ((pa.x | pb.x) & 2u) != 0u) { return false; }
    let ba = shape_body[a];
    let bb = shape_body[b];
    if ((ba < 0 && bb < 0) || (ba >= 0 && ba == bb)) { return false; }
    let wa = bitcast<i32>(pa.w);
    let wb = bitcast<i32>(pb.w);
    if (wa >= 0 && wb >= 0 && wa != wb) { return false; }
    if ((pa.y & pb.z) == 0u || (pb.y & pa.z) == 0u) { return false; }
    if (shape_geom_kind[a] == KIND_PLANE && shape_geom_kind[b] == KIND_PLANE) { return false; }
    let key = vec2<u32>(min(a, b), max(a, b));
    var lo = 0u;
    var hi = u._pad0;
    while (lo < hi) {
        let mid = lo + (hi - lo) / 2u;
        let value = collision_exclusions[mid];
        if (all(value == key)) { return false; }
        if (value.x < key.x || (value.x == key.x && value.y < key.y)) { lo = mid + 1u; }
        else { hi = mid; }
    }
    return true;
}
