
@compute @workgroup_size(64)
fn resident_gjk(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.pair_count) { return; }
    overlap[i] = 0u;
    if (atomicLoad(&resident_status[0]) != 0u) { return; }
    let count = resident_count[0];
    if (count > u.pair_count) {
        if (i == 0u) { atomicOr(&resident_status[0], FAULT_CAPACITY); }
        return;
    }
    if (i >= count || !resident_pair_valid(i)) { return; }
    if (!resident_pair_allowed(i)) { return; }
    if (!resident_shape_valid(pair_a[i]) || !resident_shape_valid(pair_b[i])) { return; }
    let ka = shape_geom_kind[pair_a[i]];
    let kb = shape_geom_kind[pair_b[i]];
    if (ka > 6u || kb > 6u) {
        overlap[i] = 2u;
        atomicOr(&resident_status[0], FAULT_GJK);
        return;
    }
    if (ka == KIND_PLANE || kb == KIND_PLANE) {
        overlap[i] = 3u;
        return;
    }
    if ((ka == KIND_SPHERE && kb == KIND_BOX) || (kb == KIND_SPHERE && ka == KIND_BOX)) {
        overlap[i] = 4u;
        return;
    }
    if (ka == KIND_SPHERE && kb == KIND_SPHERE) {
        overlap[i] = 5u;
        return;
    }
    if (ka == KIND_CAPSULE && kb == KIND_CAPSULE) {
        overlap[i] = 6u;
        return;
    }
    gjk_pair(i);
    if (overlap[i] == 2u) {
        atomicOr(&resident_status[0], FAULT_GJK);
    }
    if (overlap[i] == 1u) {
        for (var k = 0u; k < 12u; k += 1u) {
            if (!finite4(gjk_simplex[i * 12u + k])) {
                overlap[i] = 2u;
                atomicOr(&resident_status[0], FAULT_NONFINITE);
                return;
            }
        }
    }
}
