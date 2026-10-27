var<workgroup> represented_start:array<vec2<f32>,32>;

fn represented_momentum(i:u32)->vec2<f32> {
    let n=shared_params.width;
    var sum=vec2<f32>(0.0); var scale=1.0;
    for (var j=0u;j<n;j++) {
        let q=rounded(shared_q[j].x+shared_q[j].y);
        let delta=two_sum(q,-shared_free[j]);
        sum=add_product(sum,shared_mass[i*n+j],delta.x,delta.y);
        scale+=abs(shared_mass[i*n+j])*abs(q-shared_free[j]);
    }
    for (var c=0u;c<shared_params.contacts;c++) {
        if shared_contacts[c].soil_mass.w==0.0 { continue; }
        let k=-shared_outputs[c].machine_impulse_normal.xyz;
        let row=shared_rows[c*n+i].xyz;
        for (var axis=0u;axis<3u;axis++) { sum=add_product(sum,row[axis],k[axis],0.0); }
        scale+=dot(abs(row),abs(k));
    }
    return vec2<f32>(rounded(sum.x+sum.y),scale);
}

fn represented_active(c:u32)->bool {
    return shared_contacts[c].soil_mass.w>0.0
        && dot(-shared_outputs[c].machine_impulse_normal.xyz,normalize(shared_contacts[c].normal_gap.xyz))>0.0;
}

fn shared_certify()->bool {
    if shared_certify_raw() { return true; }
    if atomicLoad(&shared_status.fault)!=0u { return false; }
    let n=shared_params.width;
    var worst=1.0; var selected=n; var residual=0.0;
    for (var i=0u;i<n;i++) {
        let value=represented_momentum(i);
        let ratio=abs(value.x)/(1e-6*value.y);
        if !all(abs(value)<=vec2<f32>(3.402823e38)) || !finite(ratio) { return false; }
        if ratio>worst { worst=ratio; selected=i; residual=value.x; }
    }
    if selected==n { return false; }
    var denominator=vec2<f32>(0.0);
    for (var c=0u;c<shared_params.contacts;c++) {
        if !represented_active(c) { continue; }
        let row=shared_rows[c*n+selected].xyz;
        denominator=add_product(denominator,shared_contacts[c].soil_mass.w,dot(row,row),0.0);
    }
    let weight=rounded(denominator.x+denominator.y);
    if !finite(weight) || weight<=0.0 { return false; }
    for (var i=0u;i<n;i++) {
        represented_start[i]=shared_q[i];
        shared_q[i]=vec2<f32>(rounded(shared_q[i].x+shared_q[i].y),0.0);
    }
    for (var c=0u;c<shared_params.contacts;c++) {
        shared_history[c].baseline_k=shared_outputs[c].machine_impulse_normal;
        shared_history[c].baseline_w=shared_outputs[c].soil_velocity_energy;
    }
    var safe=true;
    for (var c=0u;c<shared_params.contacts;c++) {
        if !represented_active(c) { continue; }
        let node=shared_contacts[c]; let mass=node.soil_mass.w;
        let delta=-(mass*residual/weight)*shared_rows[c*n+selected].xyz;
        let old=-shared_outputs[c].machine_impulse_normal.xyz;
        let wall=shared_outputs[c].wall_impulse;
        let soil=shared_outputs[c].soil_velocity_energy.xyz;
        let machine=shared_velocity(c);
        let k=old+delta; let lambda=dot(k,normalize(node.normal_gap.xyz));
        if !all(abs(k)<=vec3<f32>(3.402823e38)) || !finite(lambda) || lambda<0.0
            || length(delta)>0.25e-6*(1.0+length(old)+length(wall.xyz))
            || length(delta)/mass>0.25e-6*(1.0+length(soil)+length(machine)) { safe=false; break; }
        shared_set_impulses(c,vec4<f32>(-k,lambda),wall);
    }
    if safe && shared_certify_raw() { return true; }
    for (var i=0u;i<n;i++) { shared_q[i]=represented_start[i]; }
    for (var c=0u;c<shared_params.contacts;c++) {
        shared_outputs[c].machine_impulse_normal=shared_history[c].baseline_k;
        shared_outputs[c].soil_velocity_energy=shared_history[c].baseline_w;
    }
    return false;
}
