struct Params { count: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read> awake: array<u32>;
@group(0) @binding(1) var<storage, read> sources: array<u32>;
@group(0) @binding(2) var<storage, read_write> selected: array<u32>;
@group(0) @binding(3) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(4) var<uniform> u: Params;
@compute @workgroup_size(64)
fn select_masks(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x>=u.count || atomicLoad(&status)!=0u { return; }
    let group=sources[id.x];
    var value=1u;
    if group!=0xffffffffu { value=awake[group]; }
    selected[id.x]=value;
}
