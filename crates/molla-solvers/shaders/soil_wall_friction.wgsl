struct Params { count:u32, iterations:u32, pad0:u32, pad1:u32 }
@group(0) @binding(0) var<storage,read> inputs:array<Input>;
@group(0) @binding(1) var<storage,read_write> outputs:array<Output>;
@group(0) @binding(2) var<uniform> params:Params;
@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) id:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    rounding_slot=lane;
    if id.x>=params.count { return; }
    solve_block(inputs[id.x],params.iterations);
    outputs[id.x]=block_output;
}
