@group(0) @binding(0) var<storage,read> source:array<u32>;
@group(0) @binding(1) var<storage,read_write> snapshot:array<u32>;
@group(0) @binding(2) var<storage,read_write> status:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read_write> control:array<u32>;
@compute @workgroup_size(64)
fn copy_input(@builtin(global_invocation_id) id:vec3<u32>) {
    if control[0]==0u && atomicLoad(&status[0])==0u && id.x<arrayLength(&source) {
        snapshot[id.x]=source[id.x];
    }
}
@compute @workgroup_size(1)
fn stop() {
    if control[0]==0u && atomicLoad(&status[0])==0u { control[0]=1u; }
    if control[0]!=0u { atomicOr(&status[0],1073741824u); }
}
@compute @workgroup_size(64)
fn copy_current(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x<arrayLength(&source) { snapshot[id.x]=source[id.x]; }
}
