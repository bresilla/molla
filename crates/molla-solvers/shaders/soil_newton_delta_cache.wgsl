var<private> conditioned_delta_valid:bool=false;
var<private> conditioned_delta_key:vec4<u32>;
var<private> conditioned_delta_values:array<vec2<f32>,32>;

fn conditioned_delta_refresh(next:Six) {
    let key=bitcast<vec4<u32>>(next.k);
    if conditioned_delta_valid && all(key==conditioned_delta_key) {return;}
    for(var i=0u;i<params.width;i++) {conditioned_delta_values[i]=conditioned_delta(next,i);}
    conditioned_delta_key=key;conditioned_delta_valid=true;
}
