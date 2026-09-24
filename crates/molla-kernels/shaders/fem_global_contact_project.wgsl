struct Contact {
    normal_gap: vec4<f32>, material: vec4<f32>, ids: vec4<u32>,
    support: vec4<u32>, weights: vec4<f32>,
}
struct Params { source_rows: u32, capacity: u32, particles: u32, dofs: u32 }
@group(0) @binding(0) var<storage, read> contacts: array<Contact>;
@group(0) @binding(1) var<storage, read_write> impulses: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> u: Params;

@compute @workgroup_size(64)
fn project(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= u.capacity { return; }
    let c = contacts[id.x];
    let value = impulses[id.x];
    if !all(abs(value.xyz) <= vec3<f32>(3.402823e38)) { return; }
    if c.ids.w == 0u { impulses[id.x] = vec4<f32>(0.0); return; }
    if c.ids.w == 3u { return; }
    if c.ids.w != 1u && c.ids.w != 5u { return; }
    let normal = max(0.0,value.x);
    var tangent = vec2<f32>(0.0);
    if c.ids.w == 1u && normal > 0.0 {
        if !(c.material.x >= 0.0 && c.material.x <= 3.402823e38) { return; }
        tangent = value.yz;
        let scale = max(abs(tangent.x),abs(tangent.y));
        let normalized_length = length(tangent/select(1.0,scale,scale>0.0));
        let radius = c.material.x*normal;
        if scale > 0.0 && scale > radius/normalized_length {
            tangent = (tangent/scale)*(radius/normalized_length);
        }
    }
    impulses[id.x] = vec4<f32>(normal,tangent,value.w);
}
