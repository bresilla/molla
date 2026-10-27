@group(0) @binding(0) var<storage,read> a0:array<u32>;
@group(0) @binding(1) var<storage,read> a1:array<u32>;
@group(0) @binding(2) var<storage,read> a2:array<u32>;
@group(0) @binding(3) var<storage,read> a3:array<u32>;
@group(0) @binding(4) var<storage,read> a4:array<u32>;
@group(0) @binding(5) var<storage,read> b0:array<u32>;
@group(0) @binding(6) var<storage,read> b1:array<u32>;
@group(0) @binding(7) var<storage,read> b2:array<u32>;
@group(0) @binding(8) var<storage,read> b3:array<u32>;
@group(0) @binding(9) var<storage,read> b4:array<u32>;
@group(0) @binding(10) var<storage,read_write> summary:array<u32>;
var<workgroup> differences:atomic<u32>;
var<workgroup> first:atomic<u32>;
fn count(buffer:u32)->u32 {
    switch buffer {
        case 0u: {return arrayLength(&a0);}
        case 1u: {return arrayLength(&a1);}
        case 2u: {return arrayLength(&a2);}
        case 3u: {return arrayLength(&a3);}
        default: {return arrayLength(&a4);}
    }
}
fn word(buffer:u32,index:u32,candidate:bool)->u32 {
    switch buffer {
        case 0u: {if candidate {return b0[index];}return a0[index];}
        case 1u: {if candidate {return b1[index];}return a1[index];}
        case 2u: {if candidate {return b2[index];}return a2[index];}
        case 3u: {if candidate {return b3[index];}return a3[index];}
        default: {if candidate {return b4[index];}return a4[index];}
    }
}
@compute @workgroup_size(64)
fn compare(@builtin(local_invocation_index) lane:u32) {
    if lane==0u {atomicStore(&differences,0u);atomicStore(&first,0xffffffffu);}
    workgroupBarrier();
    for(var buffer=0u;buffer<5u;buffer++) {
        for(var i=lane;i<count(buffer);i+=64u) {
            if word(buffer,i,false)!=word(buffer,i,true) {
                atomicAdd(&differences,1u);atomicMin(&first,(buffer<<28u)|i);
            }
        }
    }
    workgroupBarrier();
    if lane==0u {
        summary[0]++;
        for(var buffer=0u;buffer<5u;buffer++) {summary[8]+=count(buffer);}
        let changed=atomicLoad(&differences);
        if changed!=0u {
            summary[1]++;summary[7]+=changed;
            if summary[2]==0u {
                let key=atomicLoad(&first);let buffer=key>>28u;let index=key&0x0fffffffu;
                summary[2]=summary[0];summary[3]=buffer;summary[4]=index;
                summary[5]=word(buffer,index,false);summary[6]=word(buffer,index,true);
            }
        }
    }
}
