fn direct_worst_relative()->f32 {
    var worst=0.0;
    for(var slot=lane;slot<status.active_count;slot+=256u) {worst=max(worst,direct_contact_relative(active_indices[slot]));}
    parts[lane]=vec2<f32>(worst,0.0);workgroupBarrier();
    if lane==0u {
        control.scalar=0.0;
        for(var i=0u;i<256u;i++) {control.scalar=max(control.scalar,parts[i].x);}
    }
    return workgroupUniformLoad(&control.scalar);
}

fn direct_direction()->bool {
    let accepted=initial_direct_direction();
    var before=3.402823e38;
    if accepted {
        before=direct_worst_relative();
        if before<=params.tolerance {return true;}
    }
    if !retained_direct_ready {return accepted;}
    var relative=0.0;var residual=0.0;var beta=0.0;var iterations=0u;var fault=0u;
    let count=max(params.contacts,1u);
    if accepted {
        if lane==0u {
            relative=status.relative;residual=status.residual;beta=status.beta;
            iterations=status.iterations;fault=atomicLoad(&status.fault);
        }
        for(var slot=lane;slot<status.active_count;slot+=256u) {
            let c=active_indices[slot];direction[count+c]=direction[c];
            direction[2u*count+c]=basis[params.iterations*count+c].z;
        }
        storageBarrier();
    } else {prepare_condensed();}
    if lane==0u {atomicStore(&condensed_failed,0u);atomicStore(&status.fault,0u);control.beta=refinement_norm;status.beta=refinement_norm;}
    storageBarrier();
    if retained_direct_solve() {
        let error=true_direction_error();
        if error<=params.tolerance {
            let after=direct_worst_relative();
            if !accepted || after<before {
                if lane==0u {status.iterations=1u;}
                return true;
            }
        }
    }
    if accepted {
        for(var slot=lane;slot<status.active_count;slot+=256u) {
            let c=active_indices[slot];direction[c]=direction[count+c];
            basis[params.iterations*count+c].z=direction[2u*count+c];
        }
        if lane==0u {
            status.relative=relative;status.residual=residual;status.beta=beta;status.iterations=iterations;
            atomicStore(&status.fault,fault);atomicStore(&condensed_failed,0u);control.beta=beta;control.relative=relative;
        }
        storageBarrier();
    }
    return accepted;
}
