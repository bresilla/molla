fn represented_fixed_momentum(i:u32)->f32 {
    var sum=vec2<f32>(0.0); let n=shared_params.width;
    for (var slot=0u;slot<shared_status.control.y;slot++) {
        let c=shared_history[slot].active_index;
        if !represented_active(c) || shared_history[c].correction.w==0.0 { continue; }
        let row=shared_rows[c*n+i].xyz; let delta=shared_history[c].correction.xyz;
        for (var axis=0u;axis<3u;axis++) { sum=add_product(sum,row[axis],delta[axis],0.0); }
    }
    return rounded(sum.x+sum.y);
}
fn represented_coupled_delta(c:u32)->vec3<f32> {
    return shared_history[c].correction.xyz;
}
fn represented_redistribute()->bool {
    let n=shared_params.width;
    for (var slot=0u;slot<shared_status.control.y;slot++) {
        shared_history[shared_history[slot].active_index].correction=vec4<f32>(0.0);
    }
    var solved=false;
    for (var attempt=0u;attempt<8u;attempt++) {
        if !represented_coupled_factor() { return false; }
        var saturated=0u;
        for (var slot=0u;slot<shared_status.control.y;slot++) {
            let c=shared_history[slot].active_index;
            if !represented_active(c) || shared_history[c].correction.w!=0.0 { continue; }
            let delta=redistributed_raw_delta(c); let size=length(delta);
            if !finite(size) { return false; }
            let old=-shared_outputs[c].machine_impulse_normal.xyz;
            let wall=shared_outputs[c].wall_impulse.xyz;
            let soil=shared_outputs[c].soil_velocity_energy.xyz;
            let machine=shared_velocity(c); let mass=shared_contacts[c].soil_mass.w;
            var step=1.0;
            if size>0.0 {
                step=min(step,0.25e-6*(1.0+length(old)+length(wall))/size);
                step=min(step,0.25e-6*mass*(1.0+length(soil)+length(machine))/size);
            }
            let normal=normalize(shared_contacts[c].normal_gap.xyz);
            let change=dot(delta,normal);
            if change<0.0 { step=min(step,-dot(old,normal)/change); }
            if !finite(step) || step<0.0 { return false; }
            if step<1.0 {
                shared_history[c].correction=vec4<f32>((0.9*step)*delta,1.0);
                saturated+=1u;
            } else { shared_history[c].correction=vec4<f32>(delta,0.0); }
        }
        if saturated==0u { solved=true; break; }
    }
    if !solved { return false; }
    var low=0.0; var high=1.0;
    for (var i=0u;i<n;i++) {
        let value=represented_momentum(i); var sum=vec2<f32>(0.0);
        for (var slot=0u;slot<shared_status.control.y;slot++) {
            let c=shared_history[slot].active_index;
            if !represented_active(c) { continue; }
            let row=shared_rows[c*n+i].xyz; let delta=shared_history[c].correction.xyz;
            for (var axis=0u;axis<3u;axis++) { sum=add_product(sum,row[axis],delta[axis],0.0); }
        }
        let slope=rounded(sum.x+sum.y); let gate=1e-6*value.y;
        if !finite(slope) || !all(abs(value)<=vec2<f32>(3.402823e38)) { return false; }
        if slope==0.0 { if abs(value.x)>gate { return false; } }
        else {
            let a=(-gate-value.x)/slope; let b=(gate-value.x)/slope;
            low=max(low,min(a,b)); high=min(high,max(a,b));
        }
    }
    if !finite(low) || !finite(high) || low>high { return false; }
    let fraction=0.5*(low+high);
    for (var slot=0u;slot<shared_status.control.y;slot++) {
        let c=shared_history[slot].active_index;
        shared_history[c].correction=vec4<f32>(fraction*shared_history[c].correction.xyz,0.0);
    }
    return true;
}
