    let seed_k=-warm_seed.machine_impulse_normal.xyz;
    let seed_w=warm_seed.wall_impulse.xyz;
    x[0]=max(0.0,dot(axes[0],seed_k));
    var seed_t=vec2<f32>(dot(axes[1],seed_k),dot(axes[2],seed_k));
    let seed_radius=norm(seed_t); let seed_cap=mu*x[0];
    if seed_radius>seed_cap { seed_t=(seed_cap/seed_radius)*seed_t; }
    x[1]=seed_t.x; x[2]=seed_t.y;
    for (var i=3u;i<6u;i++) { x[i]=max(0.0,dot(axes[i],seed_w)); }
