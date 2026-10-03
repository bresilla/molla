struct QuaternionPair { high: vec4<f32>, low: vec4<f32> }
fn product_sum(slot: u32, a: vec4<f32>, b: vec4<f32>) -> vec2<f32> {
    var total = vec2<f32>(0.0);
    for (var axis = 0u; axis < 4u; axis++) {
        let product = rounded(slot,a[axis]*b[axis]);
        let tail = fma(a[axis],b[axis],-product);
        let sum = two_sum(slot,total.x,product);
        total = two_sum(slot,sum.x,sum.y+total.y+tail);
    }
    return total;
}
fn length_pair(slot: u32, high: vec4<f32>, low: vec4<f32>) -> vec2<f32> {
    let magnitude=max(max(abs(high.x),abs(high.y)),max(abs(high.z),abs(high.w)));
    let remainder=max(max(abs(low.x),abs(low.y)),max(abs(low.z),abs(low.w)));
    let scale=max(magnitude,remainder);
    if scale==0.0 { return vec2<f32>(0.0); }
    let exponent=frexp(scale).exp;
    let scaled_high=ldexp(high,vec4<i32>(-exponent));
    let scaled_low=ldexp(low,vec4<i32>(-exponent));
    var normalized_high=vec4<f32>(0.0); var normalized_low=vec4<f32>(0.0);
    for (var axis=0u; axis<4u; axis++) {
        let sum=two_sum(slot,scaled_high[axis],scaled_low[axis]);
        normalized_high[axis]=sum.x; normalized_low[axis]=sum.y;
    }
    let square=product_sum(slot,normalized_high,normalized_high);
    let square_low=square.y+2.0*dot(normalized_high,normalized_low)+dot(normalized_low,normalized_low);
    let root=rounded(slot,sqrt(square.x));
    let root_square=rounded(slot,root*root);
    let high_residual=rounded(slot,square.x-root_square);
    let low_residual=rounded(slot,square_low-fma(root,root,-root_square));
    let root_low=rounded(slot,rounded(slot,high_residual+low_residual)/(2.0*root));
    let normalized=two_sum(slot,root,root_low);
    return ldexp(normalized,vec2<i32>(exponent));
}
fn multiply(slot: u32, a: QuaternionPair, b: QuaternionPair) -> QuaternionPair {
    let left_high = vec4<f32>(a.high.w,a.high.xyz);
    let left_low = vec4<f32>(a.low.w,a.low.xyz);
    let right_high = array<vec4<f32>,4>(
        vec4<f32>(b.high.x,b.high.w,b.high.z,-b.high.y),
        vec4<f32>(b.high.y,-b.high.z,b.high.w,b.high.x),
        vec4<f32>(b.high.z,b.high.y,-b.high.x,b.high.w),
        vec4<f32>(b.high.w,-b.high.x,-b.high.y,-b.high.z));
    let right_low = array<vec4<f32>,4>(
        vec4<f32>(b.low.x,b.low.w,b.low.z,-b.low.y),
        vec4<f32>(b.low.y,-b.low.z,b.low.w,b.low.x),
        vec4<f32>(b.low.z,b.low.y,-b.low.x,b.low.w),
        vec4<f32>(b.low.w,-b.low.x,-b.low.y,-b.low.z));
    var result: QuaternionPair;
    for (var axis = 0u; axis < 4u; axis++) {
        let product = product_sum(slot,left_high,right_high[axis]);
        let low = product.y + dot(left_high,right_low[axis]) + dot(left_low,right_high[axis]);
        let normalized = two_sum(slot,product.x,low);
        result.high[axis] = normalized.x;
        result.low[axis] = normalized.y;
    }
    return result;
}
fn normalize_pair(slot: u32, value: QuaternionPair) -> QuaternionPair {
    let square = product_sum(slot,value.high,value.high);
    let square_low = square.y + 2.0*dot(value.high,value.low);
    let inverse = rounded(slot,inverseSqrt(square.x));
    let inverse_square = rounded(slot,inverse*inverse);
    let product = rounded(slot,square.x*inverse_square);
    let tail = fma(square.x,inverse_square,-product)
        + square.x*fma(inverse,inverse,-inverse_square) + square_low*inverse_square;
    let high_residual = rounded(slot,1.0-product);
    let residual = rounded(slot,high_residual-tail);
    let correction = 0.5*inverse*residual;
    var result: QuaternionPair;
    for (var axis = 0u; axis < 4u; axis++) {
        let high = rounded(slot,value.high[axis]*inverse);
        let low = fma(value.high[axis],inverse,-high)
            + value.high[axis]*correction + value.low[axis]*inverse;
        let normalized = two_sum(slot,high,low);
        result.high[axis] = normalized.x;
        result.low[axis] = normalized.y;
    }
    return result;
}
const QUATERNION_SINE_SERIES: array<vec2<f32>,8> = array<vec2<f32>,8>(
    vec2<f32>(-0.1666666716337204,4.967053879312289e-9),
    vec2<f32>(0.008333333767950535,-4.34617203337595e-10),
    vec2<f32>(-0.00019841270113829523,2.725596874933456e-12),
    vec2<f32>(0.0000027557318844628753,3.793571224297229e-14),
    vec2<f32>(-2.5052107943679403e-8,-4.4176230446483665e-16),
    vec2<f32>(1.6059044372074283e-10,-5.352526511562726e-18),
    vec2<f32>(-7.647163609812713e-13,-1.2200710471178288e-20),
    vec2<f32>(2.8114573589663704e-15,-1.0462084739763658e-22)
);
const QUATERNION_COSINE_SERIES: array<vec2<f32>,9> = array<vec2<f32>,9>(
    vec2<f32>(-0.5,0),
    vec2<f32>(0.0416666679084301,-1.2417634698280722e-9),
    vec2<f32>(-0.0013888889225199819,3.3631094437103215e-11),
    vec2<f32>(0.000024801587642286904,-3.40699609366682e-13),
    vec2<f32>(-2.755731998149713e-7,7.575112209051195e-15),
    vec2<f32>(2.0876755879584152e-9,1.1082839809204342e-16),
    vec2<f32>(-1.147074536050896e-11,-2.372207689231238e-19),
    vec2<f32>(4.7794772561329454e-14,7.62544404448643e-22),
    vec2<f32>(-1.5619206814541513e-16,-1.5404471465941993e-24)
);
fn quaternion_scalar_add(slot: u32, a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let high=two_sum(slot,a.x,b.x);
    return two_sum(slot,high.x,high.y+a.y+b.y);
}
fn quaternion_scalar_product(slot: u32, a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    let high=rounded(slot,a.x*b.x);
    let low=fma(a.x,b.x,-high)+a.x*b.y+a.y*b.x+a.y*b.y;
    return two_sum(slot,high,low);
}
fn quaternion_sincos(slot: u32, angle: vec2<f32>) -> mat2x2<f32> {
    let turns=round(angle.x*0.63661977236758134308);
    let period=quaternion_scalar_product(slot,vec2<f32>(turns,0.0),
        vec2<f32>(1.5707963705062866,-4.371138828673793e-8));
    let argument=quaternion_scalar_add(slot,angle,-period);
    let square=quaternion_scalar_product(slot,argument,argument);
    var sp=QUATERNION_SINE_SERIES[7u];
    for (var i=7u;i>0u;i--) {
        sp=quaternion_scalar_add(slot,QUATERNION_SINE_SERIES[i-1u],quaternion_scalar_product(slot,square,sp));
    }
    var cp=QUATERNION_COSINE_SERIES[8u];
    for (var i=8u;i>0u;i--) {
        cp=quaternion_scalar_add(slot,QUATERNION_COSINE_SERIES[i-1u],quaternion_scalar_product(slot,square,cp));
    }
    var sine=quaternion_scalar_product(slot,argument,quaternion_scalar_add(slot,vec2<f32>(1.0,0.0),
        quaternion_scalar_product(slot,square,sp)));
    var cosine=quaternion_scalar_add(slot,vec2<f32>(1.0,0.0),quaternion_scalar_product(slot,square,cp));
    let quadrant=turns-4.0*floor(turns*0.25);
    let original_sine=sine;
    if quadrant==1.0 {sine=cosine;cosine=-original_sine;}
    else if quadrant==2.0 {sine=-sine;cosine=-cosine;}
    else if quadrant==3.0 {sine=-cosine;cosine=original_sine;}
    return mat2x2<f32>(sine,cosine);
}
fn axis_angle_pair(slot: u32, axis: vec3<f32>, axis_low: vec3<f32>, angle: f32, angle_low: f32) -> QuaternionPair {
    let reference=quaternion_sincos(slot,two_sum(slot,0.5*angle,0.5*angle_low));
    let sine=reference[0];
    let cosine=reference[1];
    var result = QuaternionPair(vec4<f32>(0.0),vec4<f32>(0.0));
    for (var lane = 0u; lane < 3u; lane++) {
        let high = rounded(slot,axis[lane]*sine.x);
        let low = fma(axis[lane],sine.x,-high) + axis[lane]*sine.y + axis_low[lane]*sine.x;
        let sum = two_sum(slot,high,low);
        result.high[lane] = sum.x;
        result.low[lane] = sum.y;
    }
    result.high.w = cosine.x;
    result.low.w = cosine.y;
    return result;
}
