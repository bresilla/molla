    let normal=normalize(input.normal_gap.xyz);
    shared_history[c].basis[0]=vec4<f32>(normal,0.0);
    let absn=abs(normal); var axis=vec3<f32>(1.0,0.0,0.0);
    if absn.y<absn.x && absn.y<=absn.z { axis=vec3<f32>(0.0,1.0,0.0); }
    else if absn.z<absn.x && absn.z<absn.y { axis=vec3<f32>(0.0,0.0,1.0); }
    let tangent=normalize(cross(axis,normal));
    shared_history[c].basis[1]=vec4<f32>(tangent,0.0);
    shared_history[c].basis[2]=vec4<f32>(cross(normal,tangent),0.0);
    for (var i=0u;i<3u;i++) {
        let flags=(mask>>(2u*i))&3u;
        if flags==3u { return; }
        shared_history[c].basis[3u+i]=vec4<f32>(0.0);
        if flags!=0u { shared_history[c].basis[3u+i][i]=select(1.0,-1.0,flags==2u); }
    }
    for (var i=0u;i<6u;i++) {
        for (var j=0u;j<6u;j++) {
            shared_history[c].metric[i][j]=dot(shared_history[c].basis[i].xyz,shared_history[c].basis[j].xyz)/input.soil_mass.w;
            if i<3u && j<3u { shared_history[c].metric[i][j]+=dot(shared_history[c].basis[i].xyz,a*shared_history[c].basis[j].xyz); }
        }
        shared_history[c].scale[i]=1.0;
        if shared_history[c].metric[i][i]>0.0 { shared_history[c].scale[i]=1.0/shared_history[c].metric[i][i]; }
    }
    shared_history[c].scale[1]=1.0/max(shared_history[c].metric[1][1],shared_history[c].metric[2][2]);
    shared_history[c].scale[2]=shared_history[c].scale[1];
    block_output.status.x=0u;
