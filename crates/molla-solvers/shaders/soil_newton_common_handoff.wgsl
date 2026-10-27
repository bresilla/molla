fn save_common_handoff(point:u32,phase:u32,step:u32,merit:f32,state_norm:f32,alpha:f32,backtrack:u32,damping:f32,attempt:u32) {
    if lane==0u {
        condensed[4100u]=vec2<f32>(bitcast<f32>(point),0.0);
        condensed[4101u]=vec2<f32>(bitcast<f32>(phase),bitcast<f32>(step));
        condensed[4102u]=vec2<f32>(merit,state_norm);
        condensed[4103u]=vec2<f32>(alpha,bitcast<f32>(backtrack));
        condensed[4104u]=vec2<f32>(damping,bitcast<f32>(attempt));
        condensed[4105u]=vec2<f32>(fischer_smoothing,bitcast<f32>(smoothing_stalled));
        condensed[4106u]=vec2<f32>(bitcast<f32>(select(0u,1u,conditioned_model)),bitcast<f32>(select(0u,1u,direct_linearize)));
        condensed[4107u]=vec2<f32>(bitcast<f32>(outer.stop),bitcast<f32>(outer.accepted));
        condensed[4108u]=vec2<f32>(outer.error,outer.physical);
        condensed[4109u]=vec2<f32>(bitcast<f32>(outer.releases),bitcast<f32>(status.iterations));
        for(var i=0u;i<4u;i++) {condensed[4110u+i]=vec2<f32>(bitcast<f32>(release_visited[2u*i]),bitcast<f32>(release_visited[2u*i+1u]));}
    }
    if lane<params.width {condensed[4120u+lane]=rates[lane];}
    storageBarrier();
}
