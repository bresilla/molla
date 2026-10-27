fn native_sum_residual(a:f32,b:f32)->f32 {
    let aa=abs(a);let bb=abs(b);
    if max(aa,bb)<=1208925819614629174706176.0 &&
       (min(aa,bb)==0.0 || min(aa,bb)>=8.673617379884035e-19) {
        let high=a+b;let a_larger=aa>=bb;
        let larger=select(b,a,a_larger);let smaller=select(a,b,a_larger);
        return (fma(-1.0,high,larger)+smaller)*16777216.0;
    }
    return native_sum_residual_f64(a,b);
}
