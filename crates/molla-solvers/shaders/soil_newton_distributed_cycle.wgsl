fn cycle_running()->bool {
    if lane==0u {control.stop=bitcast<u32>(condensed[2182u].x);}
    return workgroupUniformLoad(&control.stop)==0u;
}

fn cycle_save(checkpoint:f32) {
    for(var i=lane;i<2080u;i+=64u) {condensed[i]=vec2<f32>(h[2u*i],h[2u*i+1u]);}
    if lane<64u {condensed[2080u+lane]=rotations[lane];}
    if lane<33u {
        var tail=0.0;if 2u*lane+1u<65u {tail=g[2u*lane+1u];}
        condensed[2144u+lane]=vec2<f32>(g[2u*lane],tail);
    }
    if lane==0u {
        condensed[2177u]=vec2<f32>(bitcast<f32>(control.stop),bitcast<f32>(control.count));
        condensed[2178u]=vec2<f32>(control.scalar,control.beta);
        condensed[2179u]=vec2<f32>(control.magnitude,control.relative);
        condensed[2180u]=vec2<f32>(refinement_norm,checkpoint);
    }
    storageBarrier();
}

fn cycle_load() {
    for(var i=lane;i<2080u;i+=64u) {h[2u*i]=condensed[i].x;h[2u*i+1u]=condensed[i].y;}
    if lane<64u {rotations[lane]=condensed[2080u+lane];}
    if lane<33u {
        g[2u*lane]=condensed[2144u+lane].x;
        if 2u*lane+1u<65u {g[2u*lane+1u]=condensed[2144u+lane].y;}
    }
    if lane==0u {
        control=Control(bitcast<u32>(condensed[2177u].x),bitcast<u32>(condensed[2177u].y),condensed[2178u].x,condensed[2178u].y,condensed[2179u].x,condensed[2179u].y);
        refinement_norm=condensed[2180u].x;
    }
    workgroupBarrier();
}

fn cycle_finish(checkpoint:f32) {
    if lane==0u && status.relative>params.tolerance {atomicOr(&status.fault,8u);}
    cycle_save(checkpoint);
    if lane==0u {condensed[2182u].x=bitcast<f32>(1u);}
}

@compute @workgroup_size(256)
fn cycle_project(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;if !cycle_running() {return;}project_rates(0u);
    if lane<params.width {condensed[4100u+lane]=rates[lane];}
}

@compute @workgroup_size(256)
fn cycle_seed(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;
    if lane==0u {atomicStore(&status.fault,0u);status.active_count=contacts[0].boundary.x;}
    for(var c=lane;c<params.contacts;c+=64u) {
        active_indices[c]=(2u*c+7u)%params.contacts;
        var b=Block();
        for(var i=0u;i<6u;i++) {
            b.residual[i]=0.1*f32(i+1u)+0.01*f32(c%7u);
            b.direct[7u*i]=vec2<f32>(2.0+0.125*f32(c%5u),0.0);
            for(var j=0u;j<3u;j++) {b.machine[3u*i+j]=vec2<f32>(f32(i32((c+i+j*7u)%13u)-6)/64.0,f32(i32((i+j)%3u)-1)/134217728.0);}
        }
        blocks[c]=b;direction[c]=Six();states[c]=State();
        direction[max(params.contacts,1u)+c]=Six();direction[2u*max(params.contacts,1u)+c]=Six();
        basis[params.iterations*max(params.contacts,1u)+c].z=Six();
    }
    storageBarrier();prepare_linear();
    for(var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];var rhs:array<f32,6>;
        for(var i=0u;i<6u;i++) {rhs[i]=-blocks[c].residual[i];}
        vector[c]=packed(rhs);scratch[c]=packed(rhs);
    }
    storageBarrier();let initial=scratch_norm();
    if lane==0u {condensed[2180u]=vec2<f32>(initial,0.0);}
}

@compute @workgroup_size(256)
fn cycle_resident(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;
    if lane==0u {refinement_norm=condensed[2180u].x;}
    workgroupBarrier();krylov_cycle(id,params.iterations);
}

@compute @workgroup_size(256)
fn cycle_publish(@builtin(local_invocation_index) id:u32) {
    for(var c=id;c<params.contacts;c+=256u) {
        states[c]=State(direction[c],basis[params.iterations*max(params.contacts,1u)+c].z);
    }
    if id==0u {
        atomicStore(&report.fault,atomicLoad(&status.fault));report.linear_iterations=status.iterations;
        report.residual=status.residual;report.physical=status.relative;report.damping=status.beta;
    }
}
