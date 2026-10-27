fn slot_constants_enabled()->bool {
    return status.active_count>0u && status.active_count<=params.contacts/13u;
}

fn slot_constant(slot:u32,index:u32)->vec2<f32> {
    let base=status.active_count+12u*slot+2u*index;
    return bitcast<vec2<f32>>(vec2<u32>(active_indices[base],active_indices[base+1u]));
}

fn store_slot_constant(slot:u32,index:u32,value:vec2<f32>) {
    let base=status.active_count+12u*slot+2u*index;
    let words=bitcast<vec2<u32>>(value);
    active_indices[base]=words.x;active_indices[base+1u]=words.y;
}

fn build_slot_constants(slot:u32) {
    for(var i=0u;i<6u;i++) {store_slot_constant(slot,i,vec2<f32>(0.0));}
    let c=active_indices[slot];let node=contacts[c];
    let square=dot(node.normal_gap.xyz,node.normal_gap.xyz);
    if !finite4(node.normal_gap) || abs(square-1.0)>1e-4 {return;}
    let normal_length=fischer_length(node.normal_gap.xyz);var trace=vec2<f32>(0.0);
    for(var axis=0u;axis<3u;axis++) {
        store_slot_constant(slot,2u+axis,fischer_div(vec2<f32>(node.normal_gap[axis],0.0),normal_length));
        for(var i=0u;i<params.width;i++) {
            trace=rate_add(trace,rate_mul(rows[c*params.width+i][axis],vec2<f32>(response[c*params.width+i][axis],0.0)));
        }
    }
    store_slot_constant(slot,0u,normal_length);store_slot_constant(slot,1u,trace);
    let scale=1.0/node.soil_mass.w+value(trace);
    if finite(scale) && scale>0.0 {
        store_slot_constant(slot,5u,fischer_div(vec2<f32>(1.0,0.0),rate_add(fischer_div(vec2<f32>(1.0,0.0),vec2<f32>(node.soil_mass.w,0.0)),trace)));
    }
}
