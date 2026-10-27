    for (var slot=0u;slot<shared_status.control.y;slot++) {
        let c=shared_history[slot].active_index;
        let mass=shared_contacts[c].soil_mass.w;
        let p=shared_outputs[c].machine_impulse_normal.xyz;
        let w=shared_outputs[c].wall_impulse.xyz;
        var fp=vec3<f32>(0.0); var fw=vec3<f32>(0.0);
        for (var axis=0u;axis<3u;axis++) {
            let dp=two_sum(p[axis],-shared_history[c].full_start_p[axis]);
            let dw=two_sum(w[axis],-shared_history[c].full_start_w[axis]);
            fp[axis]=rounded(dp.x+dp.y); fw[axis]=rounded(dw.x+dw.y);
            if has_previous {
                let difference_p=fp[axis]-shared_history[c].full_previous_p[axis];
                let difference_w=fw[axis]-shared_history[c].full_previous_w[axis];
                numerator=add_product(numerator,difference_p/mass,fp[axis],0.0);
                numerator=add_product(numerator,difference_w/mass,fw[axis],0.0);
                denominator=add_product(denominator,difference_p/mass,difference_p,0.0);
                denominator=add_product(denominator,difference_w/mass,difference_w,0.0);
            }
        }
        shared_history[c].full_previous_p=vec4<f32>(fp,0.0);
        shared_history[c].full_previous_w=vec4<f32>(fw,0.0);
    }
