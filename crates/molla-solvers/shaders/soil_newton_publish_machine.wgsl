fn certify_machine(lane:u32) {
    if lane==0u {stop=atomicLoad(&status.fault);}
    let valid=workgroupUniformLoad(&stop)==0u;
    if lane<params.width && valid {
        let i=lane;var momentum=vec2<f32>(0.0);var momentum_scale=1.0;
        for (var j=0u;j<params.width;j++) {
            let m=mass[i*params.width+j];let dj=rate_add(q[j],-free_rates[j]);
            momentum=rate_add(momentum,rate_mul(m,dj));momentum_scale+=abs(m)*abs(value(dj));
        }
        for (var c=0u;c<params.contacts;c++) {if contacts[c].soil_mass.w>0.0 {
            let k=-outputs[c].machine_impulse_normal.xyz;var reaction=vec2<f32>(0.0);
            for (var axis=0u;axis<3u;axis++) {reaction=rate_add(reaction,rate_mul(rows[c*params.width+i][axis],vec2<f32>(k[axis],0.0)));}
            momentum=rate_add(momentum,reaction);momentum_scale+=abs(value(reaction));
        }}
        if !finite(momentum_scale) || !finite(value(momentum)) {fault(2u,0u);}
        else if abs(value(momentum))>1e-6*momentum_scale {fault(4u,0u);}
    }
    if lane==0u && valid {
        var total=vec2<f32>(0.0);var energy_scale=1.0;
        for (var i=0u;i<64u;i++) {total=rate_add(total,energy_parts[i]);energy_scale+=scales[i];}
        for (var i=0u;i<params.width;i++) {
            let di=rate_add(q[i],-free_rates[i]);
            for (var j=0u;j<params.width;j++) {
                let m=mass[i*params.width+j];let dj=rate_add(q[j],-free_rates[j]);
                let e=rate_mul(0.5*m,rate_add(product(di,q[j]),product(free_rates[i],dj)));
                total=rate_add(total,e);energy_scale+=abs(value(e));
            }
        }
        status.energy=value(total);status.scale=energy_scale;
        if !finite(status.energy) || !finite(energy_scale) {fault(2u,0u);}
        else if status.energy>1e-6*energy_scale {fault(32u,0u);}
    }
}
