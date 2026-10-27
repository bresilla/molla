@group(0) @binding(0) var<storage,read> source:array<u32>;
@group(0) @binding(1) var<storage,read_write> output_words:array<u32>;
@group(0) @binding(2) var<storage,read> status:array<u32>;
@group(0) @binding(3) var<storage,read_write> control:array<u32>;
@compute @workgroup_size(1)
fn arm() {
    control[1]=u32(control[0]==0u && status[0]==0u);
    if control[1]!=0u { control[2]+=1u; }
}
@compute @workgroup_size(64)
fn capture(@builtin(global_invocation_id) id:vec3<u32>) {
    if control[0]==0u && control[1]!=0u && status[0]!=0u && id.x<arrayLength(&source) {
        output_words[id.x]=source[id.x];
    }
}
@compute @workgroup_size(1)
fn finish() {
    if control[0]==0u && control[1]!=0u && status[0]!=0u {
        control[0]=1u; control[3]=status[0];
    }
}
