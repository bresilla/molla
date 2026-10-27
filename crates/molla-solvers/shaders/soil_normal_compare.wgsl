struct Params { kind:u32, count:u32, pad0:u32, pad1:u32 }
@group(0) @binding(0) var<storage,read> reference:array<u32>;
@group(0) @binding(1) var<storage,read> candidate:array<u32>;
@group(0) @binding(2) var<storage,read_write> counts:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read> reference_status:array<u32>;
@group(0) @binding(4) var<storage,read> candidate_status:array<u32>;
@group(0) @binding(5) var<uniform> params:Params;
@compute @workgroup_size(64)
fn compare(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x==0u && params.kind==0u { atomicAdd(&counts[0],1u); }
    if id.x>=params.count { return; }
    if params.kind!=0u && (reference_status[0]!=0u || candidate_status[0]!=0u) { return; }
    if reference[id.x]!=candidate[id.x] { atomicAdd(&counts[params.kind+1u],1u); }
}
