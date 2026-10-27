fn position_pair_product(slot: u32, a: QuaternionPair, b: QuaternionPair) -> PositionPair {
    let left_high=vec4<f32>(a.high.w,a.high.xyz);
    let left_low=vec4<f32>(a.low.w,a.low.xyz);
    let right_high=array<vec4<f32>,3>(
        vec4<f32>(b.high.x,b.high.w,b.high.z,-b.high.y),
        vec4<f32>(b.high.y,-b.high.z,b.high.w,b.high.x),
        vec4<f32>(b.high.z,b.high.y,-b.high.x,b.high.w));
    let right_low=array<vec4<f32>,3>(
        vec4<f32>(b.low.x,b.low.w,b.low.z,-b.low.y),
        vec4<f32>(b.low.y,-b.low.z,b.low.w,b.low.x),
        vec4<f32>(b.low.z,b.low.y,-b.low.x,b.low.w));
    var result: PositionPair;
    for (var axis=0u;axis<3u;axis++) {
        let product=product_sum(slot,left_high,right_high[axis]);
        let low=product.y+dot(left_high,right_low[axis])+dot(left_low,right_high[axis]);
        let normalized=two_sum(slot,product.x,low);
        result.high[axis]=normalized.x;
        result.low[axis]=normalized.y;
    }
    return result;
}
