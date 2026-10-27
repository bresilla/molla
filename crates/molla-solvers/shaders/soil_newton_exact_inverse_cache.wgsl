struct ExactInverseKey {
    direct: array<vec2<f32>,36>,
    machine: array<vec2<f32>,18>,
};
var<private> exact_inverse_key: ExactInverseKey;
var<private> exact_inverse_contact: u32;
var<private> exact_inverse_valid: bool = false;
var<private> exact_inverse_hits: u32 = 0u;
var<private> exact_inverse_builds: u32 = 0u;

fn exact_cached_inverse(c:u32)->bool {
    if params.pad0==1u && exact_inverse_valid && exact_inverse_contact==c {
        var same=true;
        for(var i=0u;i<36u;i++) {
            same=same && all(bitcast<vec2<u32>>(exact_inverse_key.direct[i])==bitcast<vec2<u32>>(blocks[c].direct[i]));
        }
        for(var i=0u;i<18u;i++) {
            same=same && all(bitcast<vec2<u32>>(exact_inverse_key.machine[i])==bitcast<vec2<u32>>(blocks[c].machine[i]));
        }
        if same {exact_inverse_hits++;return true;}
    }
    exact_inverse_valid=false;
    inverses[c]=Inverse();
    exact_inverse_builds++;
    let accepted=build_inverse(c);
    if params.pad0==1u && accepted {
        exact_inverse_contact=c;
        exact_inverse_key=ExactInverseKey(blocks[c].direct,blocks[c].machine);
        exact_inverse_valid=true;
    }
    return accepted;
}
