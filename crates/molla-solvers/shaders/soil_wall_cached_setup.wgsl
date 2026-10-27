    block_output=Output(vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),vec4<f32>(0.0),vec4<u32>(1u,0u,0u,0u));
    if !all(abs(input.machine_mu.xyz)<=vec3<f32>(3.402823e38)) { return; }
    let a=mat3x3<f32>(input.a0.xyz,input.a1.xyz,input.a2.xyz);
    for (var i=0u;i<6u;i++) { axes[i]=shared_history[cache_index].basis[i].xyz; }
    h=shared_history[cache_index].metric;
    rho=shared_history[cache_index].scale;
    mu=input.machine_mu.w;
    point_inverse=a; free_soil=input.soil_mass.xyz; free_machine=input.machine_mu.xyz;
    soil_mass=input.soil_mass.w; contact_gap=input.normal_gap.w;
    for (var i=0u;i<6u;i++) {
        b[i]=dot(axes[i],input.soil_mass.xyz);
        if i<3u { b[i]-=dot(axes[i],input.machine_mu.xyz); }
        if i==0u { b[i]+=input.normal_gap.w; }
    }
