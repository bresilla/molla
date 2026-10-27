@group(0) @binding(0) var<storage, read> machine_status: array<u32>;
@group(0) @binding(1) var<storage, read_write> soil_status: atomic<u32>;
@compute @workgroup_size(1)
fn guard() {
    if machine_status[0] != 0u { atomicOr(&soil_status, 1u); }
}
