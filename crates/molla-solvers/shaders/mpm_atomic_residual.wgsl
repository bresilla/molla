fn native_sum_residual(a:f32, b:f32)->f32 {
    let high=a+b;
    let a_larger=abs(a)>=abs(b);
    let larger=select(b,a,a_larger);
    let smaller=select(a,b,a_larger);
    return f32((fma(f64(-1.0),f64(high),f64(larger))+f64(smaller))*f64(16777216.0));
}
