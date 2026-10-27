struct Params { ranges:vec4<u32>, gas:vec4<f32>, shape:vec4<f32> }
@group(0) @binding(0) var<storage,read> positions:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> hubs:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> faces:array<vec4<u32>>;
@group(0) @binding(3) var<uniform> params:Params;
@group(0) @binding(4) var<storage,read_write> result:array<vec4<f32>>;
fn finite(x:f32)->bool {return abs(x)<=3.402823e38;}
fn finite3(x:vec3<f32>)->bool {return all(abs(x)<=vec3<f32>(3.402823e38));}
fn point(i:u32,hub:vec3<f32>)->vec3<f32> {return positions[params.ranges.x+i].xyz-hub;}
@compute @workgroup_size(1)
fn cavity() {
    result[0]=vec4<f32>(0.0);
    let hub=hubs[params.ranges.z].xyz;
    if !finite3(hub) {return;}
    var volume=0.0;var correction=0.0;
    for(var i=0u;i<params.ranges.w;i++) {
        let face=faces[i];var a=vec3<f32>(0.0);var b=vec3<f32>(0.0);var c=vec3<f32>(0.0);
        if face.w==0u {a=point(face.x,hub);b=point(face.y,hub);c=point(face.z,hub);}
        else {
            a=vec3<f32>(0.0,0.0,select(-0.5,0.5,face.w==2u)*params.shape.x-hub.z);
            b=point(face.x,hub);c=point(face.y,hub);
        }
        if !finite3(a) || !finite3(b) || !finite3(c) {return;}
        let term=(a.x*(b.y*c.z-b.z*c.y)-a.y*(b.x*c.z-b.z*c.x)+a.z*(b.x*c.y-b.y*c.x))/6.0;
        let corrected=term-correction;let next=volume+corrected;
        correction=(next-volume)-corrected;volume=next;
    }
    if !finite(volume) {return;}
    volume=max(abs(volume),1e-9);
    var pressure=0.0;
    if params.gas.x>0.0 && params.gas.y>0.0 {
        pressure=clamp(params.gas.x*pow(params.gas.y/max(volume,1e-12),params.gas.z)*params.gas.w,0.0,50.0*max(params.gas.x,1e4));
    }
    let force=pressure*params.shape.y;
    if !finite(pressure) || !finite(force) {return;}
    result[0]=vec4<f32>(volume,pressure,force,1.0);
}
