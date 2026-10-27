var<workgroup> coop_active:vec2<u32>;
var<workgroup> coop_velocity:array<f32,3>;

fn coop_local_sweep() {
    let n=shared_params.width;
    for (var base=0u;base<shared_params.contacts;base+=64u) {
        if coop_lane==0u {
            coop_active=vec2<u32>(0u);
            for (var offset=0u;offset<64u && base+offset<shared_params.contacts;offset++) {
                if shared_contacts[base+offset].soil_mass.w!=0.0 { coop_active[offset/32u]|=1u<<(offset%32u); }
            }
        }
        let active_mask=workgroupUniformLoad(&coop_active);
        for (var offset=0u;offset<64u;offset++) {
        if (active_mask[offset/32u]&(1u<<(offset%32u)))==0u { continue; }
        let c=base+offset;
        if coop_lane<n {
            let old_k=-shared_outputs[c].machine_impulse_normal.xyz;
            for (var axis=0u;axis<3u;axis++) {
                shared_q[coop_lane]=add_product(shared_q[coop_lane],shared_response[c*n+coop_lane][axis],old_k[axis],0.0);
            }
        }
        workgroupBarrier();
        if coop_lane<3u {
            var sum=vec2<f32>(0.0);
            for (var i=0u;i<n;i++) {
                sum=add_product(sum,shared_rows[c*n+i][coop_lane],shared_q[i].x,shared_q[i].y);
            }
            coop_velocity[coop_lane]=rounded(sum.x+sum.y);
        }
        workgroupBarrier();
        if coop_lane==0u {
            var input=shared_inputs[c];
            input.machine_mu=vec4<f32>(coop_velocity[0],coop_velocity[1],coop_velocity[2],input.machine_mu.w);
            shared_inputs[c].machine_mu=input.machine_mu;
            cache_index=c;
            solve_block(input,shared_params.iterations);
            if block_output.status.x!=0u {
                shared_fault(64u,c+1u,block_output.status.x); coop_ready=0u;
            } else { shared_outputs[c]=block_output; }
        }
        storageBarrier();
        if workgroupUniformLoad(&coop_ready)==0u { return; }
        if coop_lane<n {
            let p=shared_outputs[c].machine_impulse_normal.xyz;
            for (var axis=0u;axis<3u;axis++) {
                shared_q[coop_lane]=add_product(shared_q[coop_lane],shared_response[c*n+coop_lane][axis],p[axis],0.0);
            }
        }
        workgroupBarrier();
        }
    }
}
