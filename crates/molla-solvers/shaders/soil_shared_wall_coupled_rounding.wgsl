var<workgroup> represented_gram:array<f32,1024>;
var<workgroup> represented_rhs:array<f32,32>;
var<workgroup> represented_scale:array<f32,32>;
var<workgroup> represented_order:array<u32,32>;
var<workgroup> represented_weights:array<f32,32>;

fn represented_coupled_factor()->bool {
    let n=shared_params.width;
    for (var i=0u;i<n;i++) {
        let value=represented_momentum(i);
        represented_rhs[i]=value.x-clamp(value.x,-0.5e-6*value.y,0.5e-6*value.y);
        represented_order[i]=i; represented_weights[i]=0.0;
        for (var j=0u;j<=i;j++) {
            var sum=vec2<f32>(0.0);
            for (var slot=0u;slot<shared_status.control.y;slot++) {
                let c=shared_history[slot].active_index;
                if !represented_active(c) { continue; }
                let value=dot(shared_rows[c*n+i].xyz,shared_rows[c*n+j].xyz);
                sum=add_product(sum,shared_contacts[c].soil_mass.w,value,0.0);
            }
            let entry=rounded(sum.x+sum.y);
            if !finite(entry) { return false; }
            represented_gram[i*n+j]=entry; represented_gram[j*n+i]=entry;
        }
        let diagonal=represented_gram[i*n+i];
        if diagonal<0.0 { return false; }
        represented_scale[i]=1.0;
        if diagonal>0.0 { represented_scale[i]=sqrt(diagonal); }
    }
    return represented_factor_solve();
}

fn represented_factor_solve()->bool {
    let n=shared_params.width;
    for (var i=0u;i<n;i++) {
        represented_rhs[i]/=represented_scale[i];
        for (var j=0u;j<n;j++) {
            represented_gram[i*n+j]=represented_gram[i*n+j]/represented_scale[i]/represented_scale[j];
            if !finite(represented_gram[i*n+j]) { return false; }
        }
    }
    var rank=0u;
    for (var k=0u;k<n;k++) {
        var p=k;
        for (var i=k+1u;i<n;i++) {
            if represented_gram[i*n+i]>represented_gram[p*n+p] { p=i; }
        }
        if represented_gram[p*n+p]<=1e-6 { break; }
        for (var j=0u;j<n;j++) {
            let value=represented_gram[k*n+j];
            represented_gram[k*n+j]=represented_gram[p*n+j]; represented_gram[p*n+j]=value;
        }
        for (var i=0u;i<n;i++) {
            let value=represented_gram[i*n+k];
            represented_gram[i*n+k]=represented_gram[i*n+p]; represented_gram[i*n+p]=value;
        }
        let rhs=represented_rhs[k]; represented_rhs[k]=represented_rhs[p]; represented_rhs[p]=rhs;
        let order=represented_order[k]; represented_order[k]=represented_order[p]; represented_order[p]=order;
        let pivot=represented_gram[k*n+k];
        for (var j=0u;j<n;j++) { represented_gram[k*n+j]/=pivot; }
        represented_rhs[k]/=pivot;
        for (var i=0u;i<n;i++) {
            if i==k { continue; }
            let multiplier=represented_gram[i*n+k];
            for (var j=0u;j<n;j++) {
                represented_gram[i*n+j]=fma(-multiplier,represented_gram[k*n+j],represented_gram[i*n+j]);
            }
            represented_rhs[i]=fma(-multiplier,represented_rhs[k],represented_rhs[i]);
        }
        rank=k+1u;
    }
    for (var i=0u;i<rank;i++) {
        let original=represented_order[i];
        let value=represented_rhs[i]/represented_scale[original];
        if !finite(value) { return false; }
        represented_weights[original]=value;
    }
    return rank>0u;
}

fn represented_coupled_delta(c:u32)->vec3<f32> {
    let n=shared_params.width; var delta=vec3<f32>(0.0);
    for (var axis=0u;axis<3u;axis++) {
        var sum=vec2<f32>(0.0);
        for (var i=0u;i<n;i++) { sum=add_product(sum,shared_rows[c*n+i][axis],represented_weights[i],0.0); }
        delta[axis]=-shared_contacts[c].soil_mass.w*rounded(sum.x+sum.y);
    }
    return delta;
}

fn represented_coupled_step()->f32 {
    var step=1.0;
    for (var slot=0u;slot<shared_status.control.y;slot++) {
        let c=shared_history[slot].active_index;
        if !represented_active(c) { continue; }
        let delta=represented_coupled_delta(c); let size=length(delta);
        if !finite(size) { return 0.0; }
        if size==0.0 { continue; }
        let old=-shared_outputs[c].machine_impulse_normal.xyz;
        let wall=shared_outputs[c].wall_impulse.xyz;
        let soil=shared_outputs[c].soil_velocity_energy.xyz;
        let machine=shared_velocity(c); let mass=shared_contacts[c].soil_mass.w;
        step=min(step,0.25e-6*(1.0+length(old)+length(wall))/size);
        step=min(step,0.25e-6*mass*(1.0+length(soil)+length(machine))/size);
        let normal=normalize(shared_contacts[c].normal_gap.xyz);
        let change=dot(delta,normal);
        if change<0.0 { step=min(step,-dot(old,normal)/change); }
    }
    if step<1.0 { step*=0.9; }
    return step;
}
