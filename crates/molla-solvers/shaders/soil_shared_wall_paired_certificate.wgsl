    for (var i=0u;i<n;i++) {
        let qi=shared_q[i];
        if !all(abs(qi)<=vec2<f32>(3.402823e38)) || !finite(qi.x+qi.y) { shared_fault(4u,0u,0u); return false; }
        var momentum=vec2<f32>(0.0); var scale=1.0;
        for (var j=0u;j<n;j++) {
            let qj=shared_q[j];
            let delta=add_product(qj,-1.0,shared_free[j].x,shared_free[j].y);
            momentum=add_product(momentum,shared_mass[i*n+j],delta.x,delta.y);
            scale+=abs(shared_mass[i*n+j])*abs(rounded(delta.x+delta.y));
            let delta_i=add_product(qi,-1.0,shared_free[i].x,shared_free[i].y);
            var product=add_product(vec2<f32>(0.0),delta_i.x,qj.x,qj.y);
            product=add_product(product,delta_i.y,qj.x,qj.y);
            product=add_product(product,shared_free[i].x,delta.x,delta.y);
            product=add_product(product,shared_free[i].y,delta.x,delta.y);
            let e=add_product(vec2<f32>(0.0),0.5*shared_mass[i*n+j],product.x,product.y);
            energy=add_product(energy,1.0,e.x,e.y); energy_scale+=abs(rounded(e.x+e.y));
        }
        for (var slot=0u;slot<shared_status.control.y;slot++) {
            let c=shared_history[slot].active_index;
            if shared_contacts[c].soil_mass.w==0.0 { continue; }
            let k=-shared_outputs[c].machine_impulse_normal.xyz;
            var contact=vec2<f32>(0.0);
            for (var axis=0u;axis<3u;axis++) { contact=add_product(contact,shared_rows[c*n+i][axis],k[axis],0.0); }
            momentum=add_product(momentum,1.0,contact.x,contact.y);
            scale+=abs(rounded(contact.x+contact.y));
        }
        if !finite(scale) || !finite(momentum.x+momentum.y) { shared_fault(4u,0u,0u); return false; }
        if abs(momentum.x+momentum.y)>1e-6*scale { return false; }
    }
