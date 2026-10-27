var<private> reduction_subgroup_lane:u32;
var<private> reduction_subgroup_width:u32;

fn reduce_parts()->vec2<f32> {
    if params.pad0==0u || reduction_subgroup_width!=32u {return reduce_parts_reference();}
    workgroupBarrier();
    let selected=subgroupAny(lane==0u);
    if selected {
        let k=reduction_subgroup_lane;
        let a=rate_add(parts[k],parts[k+128u]);
        let b=rate_add(parts[k+32u],parts[k+160u]);
        let c=rate_add(parts[k+64u],parts[k+192u]);
        let d=rate_add(parts[k+96u],parts[k+224u]);
        var total=rate_add(rate_add(a,c),rate_add(b,d));
        for(var stride=16u;stride>0u;stride/=2u) {
            let other=subgroupShuffleDown(total,stride);
            if k<stride {total=rate_add(total,other);}
        }
        if k==0u {parts[0]=total;}
    }
    workgroupBarrier();
    return parts[0];
}
