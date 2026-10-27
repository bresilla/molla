fn coordinate_round(coord: u32, value: f32) -> f32 {
    atomicStore(&position_rounding_scratch[position_rounding_lane],bitcast<u32>(value));
    let result=bitcast<f32>(atomicAdd(&position_rounding_scratch[position_rounding_lane],0u));
    atomicStore(&coordinate_error[coord],bitcast<u32>(result));
    return result;
}
