fn conditioned_contact_constants(c:u32) {
    let node=contacts[c];let offset=4160u+54u*c;
    let normal_length=fischer_length(node.normal_gap.xyz);var trace=vec2<f32>(0.0);
    // Slots 18..22 hold normal length, mobility scale and paired unit normal.
    for(var axis=0u;axis<3u;axis++) {
        condensed[offset+20u+axis]=fischer_div(vec2<f32>(node.normal_gap[axis],0.0),normal_length);
        for(var i=0u;i<params.width;i++) {
            trace=rate_add(trace,rate_mul(rows[c*params.width+i][axis],vec2<f32>(response[c*params.width+i][axis],0.0)));
        }
    }
    condensed[offset+18u]=normal_length;
    condensed[offset+19u]=fischer_div(vec2<f32>(1.0,0.0),rate_add(fischer_div(vec2<f32>(1.0,0.0),vec2<f32>(node.soil_mass.w,0.0)),trace));
}
