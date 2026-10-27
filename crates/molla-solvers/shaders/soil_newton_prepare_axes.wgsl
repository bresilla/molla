var<workgroup> prepare_state:array<atomic<u32>,16>;
var<workgroup> prepare_certificate:array<atomic<u32>,16>;

fn axis_certified(c:u32,axis:u32)->bool {
    for(var i=0u;i<params.width;i++) {
        let error=component_error(c,i,axis);
        if error<0.0 || error>1e-6 {return false;}
    }
    return true;
}

fn axis_initial_response(c:u32,axis:u32)->bool {
    let n=params.width;
    for(var i=0u;i<n;i++) {
        var sum=vec2<f32>(rows[c*n+i][axis],0.0);
        for(var j=0u;j<i;j++) {sum=add_nonzero_term(sum,-lower[i*n+j],response[c*n+j][axis]);}
        response[c*n+i][axis]=value(sum)/lower[i*n+i];
    }
    for(var k=0u;k<n;k++) {
        let i=n-1u-k;var sum=vec2<f32>(response[c*n+i][axis],0.0);
        for(var j=i+1u;j<n;j++) {sum=add_nonzero_term(sum,-lower[j*n+i],response[c*n+j][axis]);}
        let r=value(sum)/lower[i*n+i];response[c*n+i][axis]=r;
        if !finite(r) {fault(4u,c+1u);return false;}
    }
    return true;
}

fn axis_refine_response(c:u32,axis:u32,refinement:u32)->bool {
    let n=params.width;
    for(var i=0u;i<n;i++) {
        if refinement>=2u {
            let error=component_error(c,i,axis);
            if error>=0.0 && error<=1e-6 {correction[c*n+i][axis]=0.0;continue;}
        }
        var residual=vec2<f32>(rows[c*n+i][axis],0.0);
        for(var j=0u;j<n;j++) {residual=add_nonzero_term(residual,-mass[i*n+j],response[c*n+j][axis]);}
        correction[c*n+i][axis]=value(residual);
    }
    for(var i=0u;i<n;i++) {
        var sum=vec2<f32>(correction[c*n+i][axis],0.0);
        for(var j=0u;j<i;j++) {sum=add_nonzero_term(sum,-lower[i*n+j],correction[c*n+j][axis]);}
        correction[c*n+i][axis]=value(sum)/lower[i*n+i];
    }
    for(var k=0u;k<n;k++) {
        let i=n-1u-k;var sum=vec2<f32>(correction[c*n+i][axis],0.0);
        for(var j=i+1u;j<n;j++) {sum=add_nonzero_term(sum,-lower[j*n+i],correction[c*n+j][axis]);}
        correction[c*n+i][axis]=value(sum)/lower[i*n+i];
    }
    for(var i=0u;i<n;i++) {
        response[c*n+i][axis]=value(rate_sum(response[c*n+i][axis],correction[c*n+i][axis]));
        if !finite(response[c*n+i][axis]) {fault(4u,c+1u);return false;}
    }
    return true;
}

@compute @workgroup_size(64)
fn prepare(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    rate_lane=lane;let slot=lane/4u;let axis=lane%4u;let c=group.x*16u+slot;let n=params.width;
    if c<max(params.contacts,1u) {
        for(var i=0u;i<n;i++) {response[c*n+i][axis]=0.0;correction[c*n+i][axis]=0.0;}
    }
    if axis==0u {
        atomicStore(&prepare_state[slot],0u);
        if c<params.contacts && atomicLoad(&status.fault)==0u {
            let weight=contacts[c].soil_mass.w;
            if !finite(weight) || weight<0.0 {fault(1u,c+1u);}
            else if weight>0.0 {
                atomicAdd(&status.active_count,1u);var valid=true;
                for(var i=0u;i<n;i++) {
                    let row=rows[c*n+i];
                    if !all(abs(row)<=vec4<f32>(3.402823e38)) || row.w!=0.0 {valid=false;break;}
                }
                if valid {atomicStore(&prepare_state[slot],1u);} else {fault(1u,c+1u);}
            }
        }
    }
    workgroupBarrier();
    if axis<3u && atomicLoad(&prepare_state[slot])==1u {
        if !axis_initial_response(c,axis) {atomicOr(&prepare_state[slot],2u);}
    }
    workgroupBarrier();
    for(var refinement=0u;refinement<8u;refinement++) {
        if axis==0u {atomicStore(&prepare_certificate[slot],0u);}
        workgroupBarrier();
        if refinement>=2u && axis<3u && atomicLoad(&prepare_state[slot])==1u {
            if axis_certified(c,axis) {atomicOr(&prepare_certificate[slot],1u<<axis);}
        }
        workgroupBarrier();
        if axis==0u && refinement>=2u && atomicLoad(&prepare_certificate[slot])==7u {
            atomicOr(&prepare_state[slot],4u);
        }
        workgroupBarrier();
        if axis<3u && atomicLoad(&prepare_state[slot])==1u {
            if !axis_refine_response(c,axis,refinement) {atomicOr(&prepare_state[slot],2u);}
        }
        workgroupBarrier();
    }
    storageBarrier();
    if axis==0u && (atomicLoad(&prepare_state[slot])&3u)==1u {
        for(var i=0u;i<n;i++) {for(var a=0u;a<3u;a++) {
            let error=component_error(c,i,a);
            if error<0.0 {fault(4u,c+1u);return;}
            atomicMax(&status.max_residual,bitcast<u32>(error));
            if error>1e-6 {fault(4u,c+1u);return;}
        }}
    }
}
