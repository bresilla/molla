fn index_pack(value:u32)->vec2<f32> {
    return vec2<f32>(f32(value&65535u),f32(value>>16u));
}
fn index_unpack(value:vec2<f32>)->u32 {
    return u32(value.x)|(u32(value.y)<<16u);
}
fn parallel_active_indices() {
    let base=params.contacts/256u;let extra=params.contacts%256u;
    let first=base*lane+min(lane,extra);
    let end=first+base+select(0u,1u,lane<extra);
    var count=0u;
    for(var c=first;c<end;c++) {
        let mass=contacts[c].soil_mass.w;
        if !finite(mass) || mass<0.0 {atomicOr(&status.fault,1u);}
        if mass>0.0 {count++;}
    }
    parts[lane]=index_pack(count);
    workgroupBarrier();
    if lane==0u {
        var offset=0u;
        for(var i=0u;i<256u;i++) {
            let size=index_unpack(parts[i]);parts[i]=index_pack(offset);offset+=size;
        }
        status.active_count=offset;
    }
    workgroupBarrier();
    var offset=index_unpack(parts[lane]);
    for(var c=first;c<end;c++) {
        if contacts[c].soil_mass.w>0.0 {active_indices[offset]=c;offset++;}
    }
}
