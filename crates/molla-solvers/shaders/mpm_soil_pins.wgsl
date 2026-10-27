struct Params {
    minimum_and_exponent: vec4<i32>,
    dimensions_and_mask: vec4<u32>,
    count_capacity_halo: vec4<u32>,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> table: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> keys: array<u32>;
@group(0) @binding(4) var<storage, read_write> metadata: array<atomic<u32>>;

fn coordinate(bits: u32) -> vec2<i32> {
    let exponent = (bits >> 23u) & 255u;
    if exponent == 255u { return vec2<i32>(0, 2); }
    var mantissa = bits & 0x7fffffu;
    var shift = -149 - params.minimum_and_exponent.w;
    if exponent != 0u {
        mantissa |= 0x800000u;
        shift = i32(exponent) - 150 - params.minimum_and_exponent.w;
    }
    if mantissa == 0u { return vec2<i32>(0); }
    var integral = 0u;
    var fractional = false;
    if shift >= 0 {
        if shift > 7 { return vec2<i32>(0, 1); }
        integral = mantissa << u32(shift);
    } else {
        let right = u32(-shift);
        if right >= 32u {
            fractional = true;
        } else {
            integral = mantissa >> right;
            fractional = (mantissa & ((1u << right) - 1u)) != 0u;
        }
    }
    if (bits >> 31u) != 0u && fractional { integral += 1u; }
    if integral > 0x40000000u { return vec2<i32>(0, 1); }
    if (bits >> 31u) != 0u { return vec2<i32>(-i32(integral), 0); }
    return vec2<i32>(i32(integral), 0);
}

fn insert(tile: vec3<i32>) {
    let minimum = params.minimum_and_exponent.xyz;
    let dimensions = params.dimensions_and_mask.xyz;
    if any(tile < minimum) { atomicOr(&metadata[1], 1u); return; }
    let local = bitcast<vec3<u32>>(tile) - bitcast<vec3<u32>>(minimum);
    if any(local >= dimensions) { atomicOr(&metadata[1], 1u); return; }
    let key = local.x + dimensions.x * (local.y + dimensions.y * local.z);
    let stored = key + 1u;
    var hash = stored;
    hash = (hash ^ (hash >> 16u)) * 0x7feb352du;
    hash = (hash ^ (hash >> 15u)) * 0x846ca68bu;
    hash ^= hash >> 16u;
    let mask = params.dimensions_and_mask.w;
    var slot = hash & mask;
    for (var attempt = 0u; attempt < arrayLength(&table) + 32u; attempt++) {
        let claim = atomicCompareExchangeWeak(&table[slot], 0u, stored);
        if claim.exchanged {
            let output = atomicAdd(&metadata[0], 1u);
            if output < params.count_capacity_halo.y { keys[output] = key; }
            else { atomicOr(&metadata[1], 4u); }
            return;
        }
        if claim.old_value == stored { return; }
        if claim.old_value != 0u { slot = (slot + 1u) & mask; }
    }
    atomicOr(&metadata[1], 4u);
}

@compute @workgroup_size(64)
fn clear(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x < arrayLength(&table) { atomicStore(&table[gid.x], 0u); }
    if gid.x == 0u {
        atomicStore(&metadata[0], 0u);
        atomicStore(&metadata[1], 0u);
        atomicStore(&metadata[2], params.count_capacity_halo.x);
        atomicStore(&metadata[3], params.count_capacity_halo.z);
    }
}

@compute @workgroup_size(64)
fn mark(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.count_capacity_halo.x { return; }
    let position = positions[gid.x];
    let x = coordinate(position.x);
    let y = coordinate(position.y);
    let z = coordinate(position.z);
    let error = x.y | y.y | z.y;
    if error != 0 { atomicOr(&metadata[1], u32(error)); return; }
    let center = vec3<i32>(x.x, y.x, z.x);
    let halo = i32(params.count_capacity_halo.z);
    for (var dz = -halo; dz <= halo; dz++) {
        for (var dy = -halo; dy <= halo; dy++) {
            for (var dx = -halo; dx <= halo; dx++) {
                insert(center + vec3<i32>(dx, dy, dz));
            }
        }
    }
}
