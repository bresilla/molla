struct Params {
    counts:vec4<u32>, phys0:vec4<f32>, terrain_velocity:vec4<f32>,
    terrain_origin:vec4<f32>, terrain_dims:vec4<u32>, soil0:vec4<f32>,
}
struct Sample {
    normal_height:vec4<f32>, velocity_friction:vec4<f32>,
    material:vec4<f32>, status:vec4<u32>,
}
@group(0) @binding(0) var<storage,read> positions:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> terrain_heights:array<f32>;
@group(0) @binding(2) var<storage,read> terrain_friction:array<f32>;
@group(0) @binding(3) var<storage,read> terrain_water_film:array<f32>;
@group(0) @binding(4) var<storage,read> terrain_sinkage:array<f32>;
@group(0) @binding(5) var<storage,read> terrain_velocity_map:array<vec4<f32>>;
@group(0) @binding(6) var<storage,read_write> samples:array<Sample>;
@group(0) @binding(7) var<uniform> u:Params;
fn finite(v:f32)->bool {return abs(v)<=3.402823e38;}
fn coordinates(p:vec2<f32>)->bool {
    let q=(p-u.terrain_origin.xy)/u.terrain_origin.z;
    return all(abs(q)<=vec2<f32>(1e9));
}
fn valid_taps(p:vec2<f32>,material:bool)->bool {
    let q=(p-u.terrain_origin.xy)/u.terrain_origin.z;
    let base=vec2<i32>(floor(q));
    for(var z=0;i32(z)<2;z++) {for(var x=0;i32(x)<2;x++) {
        let cell=vec2<u32>(clamp(base+vec2<i32>(x,z),vec2<i32>(0),vec2<i32>(u.terrain_dims.xy)-vec2<i32>(1)));
        let at=cell.y*u.terrain_dims.x+cell.x;
        if !finite(terrain_heights[at]) {return false;}
        if material {
            if (u.terrain_dims.w&1u)!=0u && !finite(terrain_friction[at]) {return false;}
            if (u.terrain_dims.w&2u)!=0u && !finite(terrain_water_film[at]) {return false;}
            if (u.terrain_dims.w&4u)!=0u && !finite(terrain_sinkage[at]) {return false;}
            if (u.terrain_dims.w&8u)!=0u && !all(abs(terrain_velocity_map[at].xyz)<=vec3<f32>(3.402823e38)) {return false;}
        }
    }}
    return true;
}
@compute @workgroup_size(64)
fn query(@builtin(global_invocation_id) id:vec3<u32>) {
    let i=id.x;if i>=u.counts.x {return;}
    samples[i]=Sample();
    let p=positions[i].xz;let d=max(u.terrain_origin.z*0.5,1e-5);
    let probes=array<vec2<f32>,5>(p,p+vec2<f32>(d,0.0),p-vec2<f32>(d,0.0),p+vec2<f32>(0.0,d),p-vec2<f32>(0.0,d));
    for(var j=0u;j<5u;j++) {
        if !coordinates(probes[j]) {samples[i].status.x=2u;return;}
    }
    for(var j=0u;j<5u;j++) {
        if !valid_taps(probes[j],j==0u) {samples[i].status.x=3u;return;}
    }
    let nh=vec4<f32>(sample_terrain_normal(p.x,p.y),sample_terrain_height(p.x,p.y));
    let vf=vec4<f32>(sample_terrain_surface_velocity(p.x,p.y),sample_terrain_friction(p.x,p.y));
    let m=vec4<f32>(sample_terrain_water_film(p.x,p.y),sample_terrain_sinkage(p.x,p.y),u.soil0.x,0.0);
    if !all(abs(nh)<=vec4<f32>(3.402823e38)) || !all(abs(vf)<=vec4<f32>(3.402823e38)) || !all(abs(m)<=vec4<f32>(3.402823e38)) || dot(nh.xyz,nh.xyz)<0.5 {
        samples[i].status.x=3u;return;
    }
    samples[i]=Sample(nh,vf,m,vec4<u32>(1u,0u,0u,0u));
}
