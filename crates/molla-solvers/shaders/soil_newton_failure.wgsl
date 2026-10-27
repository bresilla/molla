const METHOD_TAG:u32=0u;
const CERTIFICATE_TAG:u32=1u;
struct Params { width:u32, contacts:u32, newton:u32, krylov:u32 }
@group(0) @binding(0) var<storage,read> mass:array<u32>;
@group(0) @binding(1) var<storage,read> free:array<u32>;
@group(0) @binding(2) var<storage,read> rows:array<u32>;
@group(0) @binding(3) var<storage,read> response:array<u32>;
@group(0) @binding(4) var<storage,read> contacts:array<u32>;
@group(0) @binding(5) var<storage,read> states:array<u32>;
@group(0) @binding(6) var<storage,read> report:array<u32>;
@group(0) @binding(7) var<storage,read> publication:array<u32>;
@group(0) @binding(8) var<storage,read> preparation:array<u32>;
@group(0) @binding(9) var<storage,read> upstream_common:array<u32>;
@group(0) @binding(10) var<storage,read_write> packet:array<u32>;
@group(0) @binding(11) var<uniform> params:Params;
@compute @workgroup_size(64)
fn capture(@builtin(global_invocation_id) id:vec3<u32>) {
    if upstream_common[0]!=0u || preparation[0]!=0u {return;}
    let c=id.x;let n=params.width;
    if report[0]==0u {if c==0u {packet[0]=0u;}return;}
    if c>=max(params.contacts,1u) {return;}
    if c==0u {
        packet[0]=1u;packet[1]=n;packet[2]=params.contacts;packet[3]=preparation[0];
        packet[4]=upstream_common[0];packet[5]=params.newton;packet[6]=params.krylov;
        packet[7]=METHOD_TAG;
        for(var i=8u;i<16u;i++) {packet[i]=0u;}
        packet[10]=CERTIFICATE_TAG;
        for(var i=0u;i<8u;i++) {packet[16u+i]=report[i];packet[24u+i]=publication[i];}
        for(var i=0u;i<n*n;i++) {packet[32u+i]=mass[i];}
        for(var i=0u;i<2u*n;i++) {packet[32u+n*n+i]=free[i];}
    }
    let base=32u+n*n+2u*n+c*(8u*n+32u);
    for(var i=0u;i<4u*n;i++) {
        packet[base+i]=rows[c*4u*n+i];packet[base+4u*n+i]=response[c*4u*n+i];
    }
    for(var i=0u;i<16u;i++) {
        packet[base+8u*n+i]=contacts[c*16u+i];packet[base+8u*n+16u+i]=states[c*16u+i];
    }
}
