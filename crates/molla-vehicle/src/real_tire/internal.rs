//! Shared crate-internal helpers used across the `real_tire` modules.
//!
//! These were originally inline in `real_tire.rs`; they live here so the
//! split modules (`brush`, `transient`, `contact`, `rig`, `reports`,
//! `fem`, etc.) can import them from one place.

pub(crate) fn finite_or(value: f64, fallback: f64) -> f64 {
    if value.is_finite() { value } else { fallback }
}

pub(crate) fn finite_positive_or(value: f64, fallback: f64, floor: f64) -> f64 {
    let v = finite_or(value, fallback);
    if v > floor { v } else { fallback.max(floor) }
}

pub(crate) fn finite_nonnegative(value: f64) -> f64 {
    finite_or(value, 0.0).max(0.0)
}

pub(crate) fn finite_vec3(v: [f64; 3]) -> bool {
    v.iter().all(|x| x.is_finite())
}

pub(crate) fn sanitize_vec3(v: [f64; 3], fallback: [f64; 3]) -> [f64; 3] {
    if finite_vec3(v) { v } else { fallback }
}

pub(crate) fn smoothstep01(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub(crate) fn csv_token(value: &str) -> String {
    value
        .chars()
        .map(|c| match c {
            ',' | '\n' | '\r' | '\t' => '_',
            _ => c,
        })
        .collect()
}

pub(crate) fn bool_metric(value: bool) -> f64 {
    if value { 1.0 } else { 0.0 }
}

pub(crate) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(crate) fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub(crate) fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn scale(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

pub(crate) fn normalize(v: [f64; 3]) -> [f64; 3] {
    let l = dot(v, v).sqrt();
    if l > 1.0e-12 {
        [v[0] / l, v[1] / l, v[2] / l]
    } else {
        [1.0, 0.0, 0.0]
    }
}

pub(crate) fn csv_optional_f64(value: Option<f64>) -> String {
    value
        .filter(|v| v.is_finite())
        .map(|v| format!("{v:.6}"))
        .unwrap_or_default()
}

pub(crate) fn csv_optional_vec3(value: Option<[f64; 3]>, index: usize) -> String {
    value
        .and_then(|v| v.get(index).copied())
        .filter(|v| v.is_finite())
        .map(|v| format!("{v:.6}"))
        .unwrap_or_default()
}
