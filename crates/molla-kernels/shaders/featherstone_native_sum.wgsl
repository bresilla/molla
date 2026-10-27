fn native_two_sum(a: f32, b: f32) -> vec2<f32> {
    let high = f32(f64(a) + f64(b));
    return native_two_sum_with_high(a, b, high);
}

fn native_two_sum_with_high(a: f32, b: f32, high: f32) -> vec2<f32> {
    let larger = select(b, a, abs(a) >= abs(b));
    let smaller = select(a, b, abs(a) >= abs(b));
    let difference = fma(f64(larger), f64(1.0), -f64(high));
    let low = f32(difference + f64(smaller));
    return vec2<f32>(high, low);
}
