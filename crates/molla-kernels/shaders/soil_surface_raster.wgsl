@group(1) @binding(1) var<storage, read> edges: array<f32>;
var<workgroup> top_values: array<f32,64>;
var<workgroup> count_values: array<u32,64>;
var<workgroup> outside_values: array<u32,64>;
var<workgroup> invalid_values: array<u32,64>;

fn ordered_key(v:f32)->u32 {
    let bits=bitcast<u32>(v);
    if (bits&0x7fffffffu)==0u {return 0x80000000u;}
    return select(bits|0x80000000u,~bits,(bits&0x80000000u)!=0u);
}
fn axis_index(v:f32,offset:u32,n:u32)->u32 {
    var lo=0u;var hi=n;
    while lo+1u<hi {
        let mid=lo+(hi-lo)/2u;
        if ordered_key(v)<ordered_key(edges[offset+mid]) {hi=mid;} else {lo=mid;}
    }
    return lo;
}
fn cell_index(p:vec4<f32>)->u32 {
    return axis_index(p.z,u.dims.y+1u,u.dims.z)*u.dims.y+axis_index(p.x,0u,u.dims.y);
}
fn finite_point(p:vec4<f32>)->bool {return all(abs(p.xyz)<=vec3<f32>(3.402823e38));}
fn inside(p:vec4<f32>)->bool {
    return ordered_key(p.x)>=ordered_key(edges[0]) && ordered_key(p.x)<ordered_key(edges[u.dims.y]) && ordered_key(p.z)>=ordered_key(edges[u.dims.y+1u]) && ordered_key(p.z)<ordered_key(edges[u.dims.y+1u+u.dims.z]);
}
@compute @workgroup_size(64)
fn reduce_cells(@builtin(workgroup_id) wg:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let cell=group_index(wg);
    var n=0u;var start=0u;
    if cell<u.dims.y*u.dims.z {n=atomicLoad(&bins[cell*3u]);start=atomicLoad(&bins[cell*3u+1u]);}
    var top=u.terrain.w;var count=0u;var outside=0u;var invalid=0u;
    for(var i=lane;i<n;i+=64u) {
        let p=positions[members[start+i]];
        if !finite_point(p) {invalid++;}
        else if !inside(p) {outside++;}
        else {top=max(top,p.y);count++;}
    }
    top_values[lane]=top;count_values[lane]=count;outside_values[lane]=outside;invalid_values[lane]=invalid;
    workgroupBarrier();
    for(var stride=32u;stride>0u;stride/=2u) {
        if lane<stride {
            top_values[lane]=max(top_values[lane],top_values[lane+stride]);
            count_values[lane]+=count_values[lane+stride];
            outside_values[lane]+=outside_values[lane+stride];
            invalid_values[lane]+=invalid_values[lane+stride];
        }
        workgroupBarrier();
    }
    if lane==0u && cell<u.dims.y*u.dims.z {
        heights[cell]=top_values[0];sinkage[cell]=f32(count_values[0]);
        surface_velocity[cell]=vec4<f32>(f32(outside_values[0]),f32(invalid_values[0]),0.0,0.0);
    }
}
@compute @workgroup_size(64)
fn reference_cells(@builtin(workgroup_id) wg:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    let cell=group_index(wg)*64u+lane;
    if cell>=u.dims.y*u.dims.z {return;}
    var top=u.terrain.w;var count=0u;var outside=0u;var invalid=0u;
    for(var i=0u;i<u.dims.x;i++) {
        let p=positions[i];
        if cell_index(p)!=cell {continue;}
        if !finite_point(p) {invalid++;}
        else if !inside(p) {outside++;}
        else {top=max(top,p.y);count++;}
    }
    heights[cell]=top;sinkage[cell]=f32(count);
    surface_velocity[cell]=vec4<f32>(f32(outside),f32(invalid),0.0,0.0);
}
