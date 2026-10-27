@group(0) @binding(0) var<storage,read> velocities:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> words:array<atomic<u32>>;
var<workgroup> minima:array<vec3<u32>,64>;
var<workgroup> signs:array<vec3<u32>,64>;

@compute @workgroup_size(64)
fn reference_reset(@builtin(local_invocation_index) lane:u32) {
    if lane<4u {
        atomicStore(&words[TAIL+lane],0x7f7fffffu);
        atomicStore(&words[TAIL+4u+lane],0u);
    }
}

@compute @workgroup_size(64)
fn reference_reduce(@builtin(global_invocation_id) id:vec3<u32>, @builtin(local_invocation_index) lane:u32) {
    var minimum=vec3<u32>(0x7f7fffffu);
    var flags=vec3<u32>(0u);
    if id.x<atomicLoad(&words[0]) {
        let value=velocities[id.x].xyz;
        for(var axis=0u;axis<3u;axis++) {
            let bits=bitcast<u32>(value[axis]);
            let magnitude=bits&0x7fffffffu;
            if magnitude<0x7f800000u {
                minimum[axis]=magnitude;
                flags[axis]=select(select(1u,2u,(bits&0x80000000u)!=0u),3u,magnitude==0u);
            } else { flags[axis]=3u; }
        }
    }
    minima[lane]=minimum;signs[lane]=flags;workgroupBarrier();
    for(var stride=32u;stride>0u;stride/=2u) {
        if lane<stride {
            minima[lane]=min(minima[lane],minima[lane+stride]);
            signs[lane]=signs[lane]|signs[lane+stride];
        }
        workgroupBarrier();
    }
    if lane<3u {
        atomicMin(&words[TAIL+lane],minima[0][lane]);
        atomicOr(&words[TAIL+4u+lane],signs[0][lane]);
    }
}
