@compute @workgroup_size(256)
fn distributed_project_forward(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;project_rates(0u);
    if lane<params.width {condensed[4100u+lane]=rates[lane];}
}

@compute @workgroup_size(256)
fn distributed_project_direction(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;project_rates(3u);
    if lane<params.width {condensed[4100u+lane]=rates[lane];}
}

fn distributed_product(group:u32,paired:bool) {
    if lane<params.width {rates[lane]=condensed[4100u+lane];}
    workgroupBarrier();
    if lane<30u {
        let local_contact=lane/3u;let axis=lane%3u;
        let slot=10u*group+local_contact;
        var sum=vec2<f32>(0.0);
        if slot<status.active_count {
            let c=active_indices[slot];
            for(var i=0u;i<params.width;i++) {sum=rate_add(sum,rate_mul(rows[c*params.width+i][axis],rates[i]));}
        }
        parts[lane]=sum;
    }
    workgroupBarrier();
    if lane<60u {
        let local_contact=lane/6u;let row=lane%6u;
        let slot=10u*group+local_contact;
        if slot<status.active_count {
            let c=active_indices[slot];var sum=vec2<f32>(0.0);
            for(var j=0u;j<6u;j++) {
                var x=vec2<f32>(component(vector[c],j),0.0);
                if paired {x=direction_pair(c,j);}
                sum=rate_add(sum,fischer_product(blocks[c].direct[6u*row+j],x));
            }
            for(var j=0u;j<3u;j++) {sum=rate_add(sum,fischer_product(blocks[c].machine[3u*row+j],parts[3u*local_contact+j]));}
            let v=value(sum);
            if row<3u {
                if paired {states[c].low.k[row]=v;}else {states[c].high.k[row]=v;}
            } else {
                if paired {states[c].low.w[row-3u]=v;}else {states[c].high.w[row-3u]=v;}
            }
            if row==0u {
                if paired {states[c].low.k.w=0.0;states[c].low.w.w=0.0;}
                else {states[c].high.k.w=0.0;states[c].high.w.w=0.0;}
            }
        }
    }
}

@compute @workgroup_size(8,8,1)
fn distributed_forward(@builtin(local_invocation_index) id:u32,@builtin(workgroup_id) group:vec3<u32>) {
    lane=id;rate_lane=id;distributed_product(group.x,false);
}

@compute @workgroup_size(8,8,1)
fn distributed_direction(@builtin(local_invocation_index) id:u32,@builtin(workgroup_id) group:vec3<u32>) {
    lane=id;rate_lane=id;distributed_product(group.x,true);
}

@compute @workgroup_size(1)
fn distributed_finish() {atomicStore(&report.fault,atomicLoad(&status.fault));}
