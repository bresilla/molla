struct Body {position:vec4<f32>,rotation:vec4<f32>}
@group(0) @binding(0) var<storage,read> particles:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> bodies:array<Body>;
@group(0) @binding(2) var<uniform> camera:mat4x4<f32>;
struct Output {@builtin(position) clip:vec4<f32>,@location(0) normal:vec3<f32>,@location(1) color:vec3<f32>}
fn rotate(q:vec4<f32>,v:vec3<f32>)->vec3<f32> {return v+2.0*cross(q.xyz,cross(q.xyz,v)+q.w*v);}
@vertex fn machine(@location(0) position:vec3<f32>,@location(1) normal:vec3<f32>,@location(2) color:vec3<f32>,@location(3) body:u32)->Output {
    let pose=bodies[body];var out:Output;
    out.clip=camera*vec4<f32>(pose.position.xyz+rotate(pose.rotation,position),1.0);
    out.normal=rotate(pose.rotation,normal);out.color=color;return out;
}
@vertex fn particle(@builtin(vertex_index) vertex:u32,@builtin(instance_index) instance:u32)->Output {
    let offsets=array<vec3<f32>,4>(vec3<f32>(1.0,1.0,1.0),vec3<f32>(-1.0,-1.0,1.0),vec3<f32>(-1.0,1.0,-1.0),vec3<f32>(1.0,-1.0,-1.0));
    let indices=array<u32,12>(0u,2u,1u,0u,1u,3u,0u,3u,2u,1u,2u,3u);
    let face=(vertex/3u)*3u;var out:Output;
    out.clip=camera*vec4<f32>(particles[instance].xyz+0.012*offsets[indices[vertex]],1.0);
    out.normal=normalize(cross(offsets[indices[face+1u]]-offsets[indices[face]],offsets[indices[face+2u]]-offsets[indices[face]]));
    out.color=vec3<f32>(0.43,0.24,0.10);return out;
}
@fragment fn fragment(input:Output)->@location(0) vec4<f32> {
    let light=0.35+0.65*abs(dot(normalize(input.normal),normalize(vec3<f32>(0.4,0.8,0.3))));
    return vec4<f32>(input.color*light,1.0);
}
