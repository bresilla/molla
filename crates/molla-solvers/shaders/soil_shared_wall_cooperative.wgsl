var<private> coop_lane:u32;
var<private> coop_start:array<vec2<f32>,32>;
var<workgroup> coop_ready:u32;
var<workgroup> coop_trial:f32;
var<workgroup> coop_rejected:u32;
var<workgroup> coop_terms:array<f32,64>;
var<workgroup> coop_merit:f32;

fn coop_rebuild_rates() {
    let n=shared_params.width; let i=coop_lane;
    if i<n {
        var sum=vec2<f32>(shared_free[i],0.0);
        for (var c=0u;c<shared_params.contacts;c++) {
            if shared_contacts[c].soil_mass.w==0.0 { continue; }
            let p=shared_outputs[c].machine_impulse_normal.xyz;
            for (var axis=0u;axis<3u;axis++) { sum=add_product(sum,shared_response[c*n+i][axis],p[axis],0.0); }
        }
        shared_q[i]=sum;
    }
    workgroupBarrier();
}
fn coop_natural_merit()->f32 {
    var value=vec2<f32>(0.0); var invalid=false;
    for (var base=0u;base<shared_params.contacts;base+=64u) {
        let c=base+coop_lane;
        var term=0.0;
        if c<shared_params.contacts { term=coop_contact_merit(c); }
        coop_terms[coop_lane]=term;
        workgroupBarrier();
        if coop_lane==0u {
            for (var offset=0u;offset<64u && base+offset<shared_params.contacts;offset++) {
                if shared_contacts[base+offset].soil_mass.w==0.0 { continue; }
                let entry=coop_terms[offset];
                if entry==3.402823e38 { invalid=true; }
                value=two_sum(value.x,value.y+entry);
            }
        }
        workgroupBarrier();
    }
    if coop_lane==0u {
        coop_merit=value.x+value.y;
        if invalid || !finite(coop_merit) { coop_merit=3.402823e38; }
    }
    return workgroupUniformLoad(&coop_merit);
}
fn coop_accelerate(has_previous:bool) {
    if coop_lane==0u { coop_trial=coop_coefficient(coop_start,has_previous); }
    let coefficient=workgroupUniformLoad(&coop_trial);
    let baseline=coop_natural_merit();
    for (var c=coop_lane;c<shared_params.contacts;c+=64u) {
        if shared_contacts[c].soil_mass.w==0.0 { continue; }
        let p=shared_outputs[c].machine_impulse_normal; let w=shared_outputs[c].wall_impulse;
        shared_history[c].baseline_k=p; shared_history[c].baseline_w=w;
        if coefficient!=0.0 {
            let trial_p=p.xyz+coefficient*(p.xyz-shared_history[c].previous_k.xyz);
            let trial_w=w.xyz+coefficient*(w.xyz-shared_history[c].previous_w.xyz);
            let lambda=dot(-trial_p,normalize(shared_contacts[c].normal_gap.xyz));
            shared_set_impulses(c,vec4<f32>(trial_p,lambda),vec4<f32>(trial_w,0.0));
        }
    }
    storageBarrier();
    if coefficient!=0.0 {
        if coop_lane==0u { shared_status.attempted+=1u; }
        coop_rebuild_rates();
        let candidate=coop_natural_merit();
        if coop_lane==0u {
            coop_rejected=select(0u,1u,candidate>=baseline);
            if coop_rejected!=0u { shared_status.rejected+=1u; }
            else { shared_status.accelerated+=1u; }
        }
        if workgroupUniformLoad(&coop_rejected)!=0u {
            for (var c=coop_lane;c<shared_params.contacts;c+=64u) {
                if shared_contacts[c].soil_mass.w==0.0 { continue; }
                shared_set_impulses(c,shared_history[c].baseline_k,shared_history[c].baseline_w);
            }
            storageBarrier();
            coop_rebuild_rates();
        }
    }
    for (var c=coop_lane;c<shared_params.contacts;c+=64u) {
        shared_history[c].previous_k=shared_history[c].baseline_k;
        shared_history[c].previous_w=shared_history[c].baseline_w;
    }
    storageBarrier();
}
@compute @workgroup_size(64)
fn shared_solve(@builtin(local_invocation_index) lane:u32) {
    coop_lane=lane; rounding_slot=lane;
    let n=shared_params.width;
    if lane==0u {
        coop_ready=0u;
        if atomicLoad(&shared_status.fault)==0u && shared_status.control.x==0u {
            for (var i=0u;i<n;i++) {
                shared_q[i]=shared_status.rates[i];
                shared_previous_f[i]=shared_status.previous_f[i];
            }
            coop_start=shared_q;
            shared_status.sweeps+=1u;
            coop_local_sweep();
            if atomicLoad(&shared_status.fault)==0u { coop_ready=1u; }
        }
    }
    if workgroupUniformLoad(&coop_ready)==0u { return; }
    storageBarrier();
    coop_rebuild_rates();
    if SHARED_ACCELERATE { coop_accelerate(shared_status.sweeps>1u); }
    if lane==0u { coop_finish(); }
}
