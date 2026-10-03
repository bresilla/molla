struct Transform { position: vec4<f32>, rotation: vec4<f32> }
struct Joint {
    metadata: vec4<u32>, axis: vec4<f32>, axis_low: vec4<f32>,
    parent_high: vec4<f32>, parent_low: vec4<f32>,
    child_high: vec4<f32>, child_low: vec4<f32>
}
@group(0) @binding(0) var<storage, read> joints: array<Joint>;
@group(0) @binding(1) var<storage, read> starts: array<u32>;
@group(0) @binding(2) var<storage, read> coordinates: array<f32>;
@group(0) @binding(3) var<storage, read> coordinate_residues: array<f32>;
@group(0) @binding(4) var<storage, read> poses: array<Transform>;
@group(0) @binding(5) var<storage, read_write> rotations: array<QuaternionPair>;
@group(0) @binding(6) var<storage, read_write> rounding: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> status: atomic<u32>;
fn rounded(slot: u32, value: f32) -> f32 {
    atomicStore(&rounding[slot],bitcast<u32>(value));
    return bitcast<f32>(atomicLoad(&rounding[slot]));
}
fn two_sum(slot: u32, a: f32, b: f32) -> vec2<f32> {
    let sum = rounded(slot,a+b);
    let virtual_b = rounded(slot,sum-a);
    let virtual_a = rounded(slot,sum-virtual_b);
    return vec2<f32>(sum,rounded(slot,(a-virtual_a)+(b-virtual_b)));
}
fn finite(value: vec4<f32>) -> bool { return all(abs(value) <= vec4<f32>(3.402823e38)); }
@compute @workgroup_size(64)
fn initialize(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= arrayLength(&poses) { return; }
    let norm = dot(poses[id.x].rotation,poses[id.x].rotation);
    if !finite(poses[id.x].rotation) || !(norm > 0.0 && norm <= 3.402823e38) { atomicOr(&status,32768u); return; }
    let rotation=normalize_pair(id.x,QuaternionPair(poses[id.x].rotation,vec4<f32>(0.0)));
    if !finite(rotation.high) || !finite(rotation.low) { atomicOr(&status,32768u); return; }
    rotations[id.x]=rotation;
}
@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x+1u >= arrayLength(&starts) || atomicLoad(&status) != 0u { return; }
    for (var index = starts[id.x]; index < starts[id.x+1u]; index++) {
        let joint = joints[index];
        let child = joint.metadata.y;
        if u32(joint.axis.w) != 0u { continue; }
        let coord = joint.metadata.w;
        let kind = joint.metadata.z;
        var local = QuaternionPair(vec4<f32>(0.0,0.0,0.0,1.0),vec4<f32>(0.0));
        if kind == 1u || kind == 7u {
            let offset = coord+select(0u,1u,kind == 7u);
            let axis = select(joint.axis.xyz,vec3<f32>(1.0,0.0,0.0),kind == 7u);
            let low = select(joint.axis_low.xyz,vec3<f32>(0.0),kind == 7u);
            local = axis_angle_pair(child,axis,low,coordinates[offset],coordinate_residues[offset]);
        } else if kind == 2u || (kind >= 4u && kind <= 6u) {
            let offset = coord+select(3u,0u,kind == 2u);
            local.high = vec4<f32>(coordinates[offset],coordinates[offset+1u],coordinates[offset+2u],coordinates[offset+3u]);
            if (u32(joint.axis_low.w)&1u) != 0u {
                local.low = vec4<f32>(coordinate_residues[offset],coordinate_residues[offset+1u],coordinate_residues[offset+2u],coordinate_residues[offset+3u]);
            }
        }
        if (u32(joint.axis_low.w)&2u) != 0u {
            local.high = vec4<f32>(-local.high.xyz,local.high.w);
            local.low = vec4<f32>(-local.low.xyz,local.low.w);
        }
        var parent = QuaternionPair(vec4<f32>(0.0,0.0,0.0,1.0),vec4<f32>(0.0));
        if joint.metadata.x != 0xffffffffu { parent = rotations[joint.metadata.x]; }
        let xp = QuaternionPair(joint.parent_high,joint.parent_low);
        let xc = QuaternionPair(vec4<f32>(-joint.child_high.xyz,joint.child_high.w),
            vec4<f32>(-joint.child_low.xyz,joint.child_low.w));
        let result = normalize_pair(child,multiply(child,multiply(child,multiply(child,parent,xp),local),xc));
        if !finite(result.high) || !finite(result.low) { atomicOr(&status,32768u); return; }
        rotations[child] = result;
    }
}
