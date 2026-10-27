@group(0) @binding(0) var<storage,read> cavities:array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> rows:array<u32>;
@group(0) @binding(2) var<storage,read_write> control:array<u32>;
@compute @workgroup_size(1)
fn admit() {
    var valid=control[0]==0u;
    for(var i=0u;i<control[13];i++) {
        let row=cavities[i];
        valid=valid && row.w==1.0 && row.x>0.0 && all(row.xyz>=vec3<f32>(0.0)) && all(abs(row.xyz)<=vec3<f32>(3.402823e38));
    }
    control[1]=select(0u,1u,valid);
    control[4]=select(0u,control[14],valid);control[5]=1u;control[6]=1u;
    control[7]=select(0u,1u,valid);control[8]=1u;control[9]=1u;
    control[10]=select(0u,control[13],valid);control[11]=1u;control[12]=1u;
    if !valid {control[0]=1u;return;}
    control[2]=min(control[2],0xfffffffeu)+1u;
    for(var i=0u;i<control[13];i++) {rows[28u*i+21u]=bitcast<u32>(cavities[i].z);}
}
