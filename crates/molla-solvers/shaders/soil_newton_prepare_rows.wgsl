var<workgroup> row_solution:array<f32,96>;
var<workgroup> row_response:array<f32,96>;
var<workgroup> row_fault:atomic<u32>;
var<workgroup> row_active:u32;
var<workgroup> row_done:u32;

fn row_solve(rhs:f32,i:u32,axis:u32)->f32 {
    let n=params.width;
    var sum=vec2<f32>(rhs,0.0);
    for(var k=0u;k<n;k++) {
        if i==k {row_solution[3u*i+axis]=value(sum)/lower[i*n+i];}
        workgroupBarrier();
        if i<n && i>k {sum=add_nonzero_term(sum,-lower[i*n+k],row_solution[3u*k+axis]);}
    }
    for(var k=0u;k<n;k++) {
        let pivot=n-1u-k;
        if i==pivot {
            sum=vec2<f32>(row_solution[3u*i+axis],0.0);
            for(var j=i+1u;j<n;j++) {sum=add_nonzero_term(sum,-lower[j*n+i],row_solution[3u*j+axis]);}
            row_solution[3u*i+axis]=value(sum)/lower[i*n+i];
        }
        workgroupBarrier();
    }
    if i<n {return row_solution[3u*i+axis];}
    return 0.0;
}

fn row_error(c:u32,i:u32,axis:u32)->f32 {
    let n=params.width;
    var residual=vec2<f32>(-rows[c*n+i][axis],0.0);
    var scale=abs(rows[c*n+i][axis]);
    for(var j=0u;j<n;j++) {
        let m=mass[i*n+j];let r=row_response[3u*j+axis];
        residual=add_nonzero_term(residual,m,r);scale+=abs(m)*abs(r);
    }
    let error=abs(value(residual))/select(scale,1.0,scale==0.0);
    return select(-1.0,error,finite(error)&&finite(scale));
}

@compute @workgroup_size(128)
fn prepare(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
    rate_lane=lane;let c=group.x;let n=params.width;let i=lane/3u;let axis=lane%3u;
    if i<n {
        response[c*n+i][axis]=0.0;correction[c*n+i][axis]=0.0;
        if axis==0u {response[c*n+i].w=0.0;correction[c*n+i].w=0.0;}
    }
    if lane==0u {
        row_active=0u;atomicStore(&row_fault,0u);
        if c<params.contacts && atomicLoad(&status.fault)==0u {
            let weight=contacts[c].soil_mass.w;
            if !finite(weight) || weight<0.0 {fault(1u,c+1u);}
            else if weight>0.0 {row_active=1u;atomicAdd(&status.active_count,1u);}
        }
    }
    if workgroupUniformLoad(&row_active)==0u {return;}
    if i<n {
        let row=rows[c*n+i];
        if !all(abs(row)<=vec4<f32>(3.402823e38)) || row.w!=0.0 {atomicOr(&row_fault,1u);}
    }
    workgroupBarrier();
    if lane==0u && atomicLoad(&row_fault)!=0u {fault(1u,c+1u);row_active=0u;}
    if workgroupUniformLoad(&row_active)==0u {return;}
    var rhs=0.0;
    if i<n {rhs=rows[c*n+i][axis];}
    var current=row_solve(rhs,i,axis);
    var delta=0.0;
    if i<n {row_response[3u*i+axis]=current;if !finite(current) {atomicOr(&row_fault,4u);}}
    workgroupBarrier();
    for(var refinement=0u;refinement<8u;refinement++) {
        if lane==0u {row_done=1u;}
        workgroupBarrier();
        var error=0.0;
        if refinement>=2u && i<n {error=row_error(c,i,axis);}
        if i<n && (refinement<2u || error<0.0 || error>1e-6) {atomicOr(&row_fault,8u);}
        workgroupBarrier();
        if lane==0u {
            let bits=atomicLoad(&row_fault);
            row_done=select(1u,0u,(bits&8u)!=0u);
            if (bits&4u)!=0u {fault(4u,c+1u);row_done=2u;}
            atomicStore(&row_fault,bits&4u);
        }
        let done=workgroupUniformLoad(&row_done);
        if done==2u {return;}
        if done==1u {break;}
        rhs=0.0;
        if i<n && (refinement<2u || error<0.0 || error>1e-6) {
            var residual=vec2<f32>(rows[c*n+i][axis],0.0);
            for(var j=0u;j<n;j++) {residual=add_nonzero_term(residual,-mass[i*n+j],row_response[3u*j+axis]);}
            rhs=value(residual);
        }
        delta=row_solve(rhs,i,axis);
        if i<n {
            current=value(rate_sum(current,delta));row_response[3u*i+axis]=current;
            if !finite(current) {atomicOr(&row_fault,4u);}
        }
        workgroupBarrier();
    }
    if i<n {
        let error=row_error(c,i,axis);
        if error<0.0 {fault(4u,c+1u);}
        else {atomicMax(&status.max_residual,bitcast<u32>(error));if error>1e-6 {fault(4u,c+1u);}}
        response[c*n+i][axis]=current;correction[c*n+i][axis]=delta;
    }
}
