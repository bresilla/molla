@group(0) @binding(0) var<storage, read> projected: array<f32>;
@group(0) @binding(1) var<storage, read> soil_status: array<u32>;
@group(0) @binding(2) var<storage, read_write> rates: array<f32>;
@group(0) @binding(3) var<storage, read_write> delta: array<f32>;
@group(0) @binding(4) var<storage, read_write> machine_status: atomic<u32>;
@group(0) @binding(5) var<uniform> params: vec4<u32>;

fn finite(x: f32) -> bool { return abs(x) <= 3.402823e38; }

@compute @workgroup_size(1)
fn publish() {
    if soil_status[0] != 0u { atomicOr(&machine_status, 32u); }
    if atomicLoad(&machine_status) != 0u { return; }
    var next_delta: array<f32, 32>;
    for (var i=0u; i<params.x; i++) {
        let change = projected[i] - rates[i];
        next_delta[i] = delta[i];
        if change != 0.0 { next_delta[i] += change; }
        if !finite(projected[i]) || !finite(rates[i]) || !finite(delta[i])
            || !finite(change) || !finite(next_delta[i]) {
            atomicOr(&machine_status, 32u); return;
        }
    }
    for (var i=0u; i<params.x; i++) {
        rates[i] = projected[i];
        delta[i] = next_delta[i];
    }
}
