fn shared_normal_residual(c:u32,soil:vec3<f32>)->vec2<f32> {
    let node=shared_contacts[c]; var machine=vec2<f32>(0.0);
    for (var i=0u;i<shared_params.width;i++) {
        machine=normal_rate_add(machine,node.normal_gap.xyz,shared_rows[c*shared_params.width+i].xyz,shared_q[i]);
    }
    return normal_rate_residual(soil,node.normal_gap.xyz,node.normal_gap.w,machine);
}
fn shared_normal_covered(c:u32,soil:vec3<f32>)->bool {
    let residual=shared_normal_residual(c,soil);
    return all(abs(residual)<=vec2<f32>(3.402823e38)) && residual.x>=-1e-6*residual.y;
}
