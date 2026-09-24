struct PatchContact {
    normal_gap: vec4<f32>, material: vec4<f32>, ids: vec4<u32>,
    support: vec4<u32>, weights: vec4<f32>,
}
@group(0) @binding(0) var<storage, read> metadata: array<f32>;
@group(0) @binding(1) var<storage, read> solution: array<f32>;
@group(0) @binding(2) var<storage, read_write> contacts: array<PatchContact>;
@group(0) @binding(3) var<storage, read_write> input_velocity: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> inverse_mass: array<f32>;

fn set_contacts(kind: u32) {
    if metadata[191] == 0.0 { return; }
    for (var row = 0u; row < 3u; row += 1u) {
        contacts[u32(metadata[187u+row])].ids.w = kind;
    }
}

@compute @workgroup_size(1)
fn disable_contacts() { set_contacts(0u); }

@compute @workgroup_size(1)
fn restore_contacts() { set_contacts(1u); }

@compute @workgroup_size(1)
fn sync_target() {
    if metadata[191] == 0.0 || solution[36] != 1.0 { return; }
    for (var i = 0u; i < 2u; i += 1u) {
        let node = u32(metadata[i]);
        var impulse = vec3<f32>(0.0);
        for (var j = 0u; j < 9u; j += 1u) {
            let change = metadata[190]*(solution[6u+j]*metadata[2]-metadata[22u+j]);
            for (var k = 0u; k < 3u; k += 1u) {
                impulse[k] += metadata[40u+j*6u+i*3u+k]*change;
            }
        }
        input_velocity[node] = vec4<f32>(input_velocity[node].xyz+inverse_mass[node]*impulse,0.0);
    }
}
