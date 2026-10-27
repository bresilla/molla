var<workgroup> ordered_total:u32;

fn coop_natural_merit()->f32 {
    let count=workgroupUniformLoad(&ordered_total);
    var value=vec2<f32>(0.0); var invalid=false;
    for (var base=0u;base<count;base+=64u) {
        let slot=base+coop_lane;
        var term=0.0;
        if slot<count { term=coop_contact_merit(shared_history[slot].active_index); }
        coop_terms[coop_lane]=term;
        workgroupBarrier();
        if coop_lane==0u {
            for (var offset=0u;offset<64u && base+offset<count;offset++) {
                let entry=coop_terms[offset];
                if entry==3.402823e38 { invalid=true; }
                value=two_sum(value.x,value.y+entry);
            }
        }
        workgroupBarrier();
    }
    if coop_lane==0u {
        coop_merit=value.x+value.y;
        if invalid || !finite(coop_merit) { coop_merit=3.402823e38; }
    }
    return workgroupUniformLoad(&coop_merit);
}
