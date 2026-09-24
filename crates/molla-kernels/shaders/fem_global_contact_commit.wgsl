struct Row { diagonal: vec4<f32>, lambda: vec4<f32> }
struct Params { source_rows: u32, capacity: u32, particles: u32, dofs: u32 }
@group(0) @binding(0) var<storage, read> schedule: array<u32>;
@group(0) @binding(1) var<storage, read> impulses: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> soft_candidate: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> rigid_candidate: array<f32>;
@group(0) @binding(4) var<storage, read> soft_terms: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> newton_status: array<u32>;
@group(0) @binding(6) var<storage, read> compact_status: array<u32>;
@group(0) @binding(7) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(8) var<storage, read_write> rows: array<Row>;
@group(0) @binding(9) var<storage, read_write> soft_velocity: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read_write> rigid_velocity: array<f32>;
@group(0) @binding(11) var<storage, read_write> input_velocity: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read_write> gate: atomic<u32>;
@group(0) @binding(13) var<uniform> u: Params;
fn finite(v: f32) -> bool { return abs(v) <= 3.402823e38; }
fn finite3(v: vec3<f32>) -> bool { return all(abs(v)<=vec3<f32>(3.402823e38)); }

@compute @workgroup_size(1)
fn initialize() {
    var faults = atomicLoad(&status)|compact_status[0];
    if newton_status[0] != 0u || newton_status[10] != 1u { faults |= 64u; }
    if schedule[0] > u.capacity { faults |= 16u; }
    atomicStore(&gate,faults);
    atomicOr(&status,faults);
}

@compute @workgroup_size(64)
fn validate(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    var faults = 0u;
    if i < u.particles {
        if !finite3(soft_candidate[i].xyz) || !finite3(input_velocity[i].xyz+soft_terms[i].xyz) { faults |= 4u; }
    }
    if i < u.dofs && !finite(rigid_candidate[i]) { faults |= 4u; }
    if i < min(schedule[0],u.capacity) {
        if !finite3(impulses[i].xyz) { faults |= 4u; }
        if schedule[i+1u] >= u.source_rows { faults |= 2u; }
    }
    atomicOr(&gate,faults);
    atomicOr(&status,faults);
}

@compute @workgroup_size(64)
fn commit(@builtin(global_invocation_id) id: vec3<u32>) {
    if atomicLoad(&gate) != 0u { return; }
    let i = id.x;
    if i < u.particles {
        soft_velocity[i] = soft_candidate[i];
        input_velocity[i] = vec4<f32>(input_velocity[i].xyz+soft_terms[i].xyz,input_velocity[i].w);
    }
    if i < u.dofs { rigid_velocity[i] = rigid_candidate[i]; }
    if i < schedule[0] { rows[schedule[i+1u]].lambda = impulses[i]; }
}
