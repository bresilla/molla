struct Contact { normal_gap: vec4<f32>, material: vec4<f32>, ids: vec4<u32>, support: vec4<u32>, weights: vec4<f32> }
struct Row { diagonal: vec4<f32>, lambda: vec4<f32> }
struct Params { source_rows: u32, capacity: u32, particles: u32, dofs: u32 }
@group(0) @binding(0) var<storage, read> original_contacts: array<Contact>;
@group(0) @binding(1) var<storage, read> schedule: array<u32>;
@group(0) @binding(2) var<storage, read> original_rows: array<Row>;
@group(0) @binding(3) var<storage, read> original_projected: array<f32>;
@group(0) @binding(4) var<storage, read> original_response: array<f32>;
@group(0) @binding(5) var<storage, read_write> original_status: atomic<u32>;
@group(0) @binding(6) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(7) var<storage, read_write> rows: array<Row>;
@group(0) @binding(8) var<storage, read_write> projected: array<f32>;
@group(0) @binding(9) var<storage, read_write> response: array<f32>;
@group(0) @binding(10) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(11) var<storage, read_write> impulses: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read_write> inverse_diagonal: array<f32>;
@group(0) @binding(13) var<uniform> u: Params;

@compute @workgroup_size(1)
fn initialize() {
    if schedule[0] > u.capacity { atomicOr(&original_status,16u); }
    if schedule[0] > u.source_rows { atomicOr(&original_status,2u); }
    atomicStore(&status,atomicLoad(&original_status));
}

@compute @workgroup_size(64)
fn compact(@builtin(global_invocation_id) id: vec3<u32>) {
    let r = id.x;
    if r >= u.capacity { return; }
    var valid = r < schedule[0] && atomicLoad(&original_status) == 0u;
    var source_row = 0u;
    if valid {
        source_row = schedule[r+1u];
        if source_row >= u.source_rows {
            atomicOr(&status,2u); atomicOr(&original_status,2u); valid = false;
        }
    }
    contacts[r] = Contact(vec4<f32>(0.0),vec4<f32>(0.0),vec4<u32>(0u),vec4<u32>(0u),vec4<f32>(0.0));
    rows[r] = Row(vec4<f32>(0.0),vec4<f32>(0.0));
    if valid { contacts[r] = original_contacts[source_row]; rows[r] = original_rows[source_row]; }
    impulses[r] = rows[r].lambda;
    for (var k = 0u; k < 3u*u.dofs; k++) {
        var a = 0.0; var b = 0.0;
        if valid { a = original_projected[source_row*3u*u.dofs+k]; b = original_response[source_row*3u*u.dofs+k]; }
        projected[r*3u*u.dofs+k] = a;
        response[r*3u*u.dofs+k] = b;
    }
}

@compute @workgroup_size(64)
fn preconditioner(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let begin = 3u*u.particles+u.dofs;
    if i >= begin+3u*u.capacity { return; }
    var value = 1.0;
    if i >= begin {
        let row = (i-begin)/3u;
        let axis = (i-begin)%3u;
        let kind = contacts[row].ids.w;
        var diagonal = 1.0;
        if kind == 3u { diagonal = rows[row].diagonal[axis]; }
        else if kind == 1u { diagonal = rows[row].diagonal[min(axis,1u)]; }
        else if kind == 5u && axis == 0u { diagonal = rows[row].diagonal.x; }
        if diagonal > 0.0 { value = 1.0/diagonal; }
    }
    inverse_diagonal[i] = value;
}
