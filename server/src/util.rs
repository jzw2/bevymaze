pub fn lin_map(a: f64, b: f64, c: f64, d: f64, x: f64) -> f64 {
    return (d - c) / (b - a) * (x - a) + c;
}

pub fn lin_map32(a: f32, b: f32, c: f32, d: f32, x: f32) -> f32 {
    return (d - c) / (b - a) * (x - a) + c;
}

/// A smooth approximation to the maximum function
/// Using the formula here https://en.wikipedia.org/wiki/Smooth_maximum#Smooth_maximum_unit
pub fn smooth_maximum_unit(a: f64, b: f64, epsilon: f64) -> f64 {
    let amb = a - b;
    return (a + b + (amb * amb + epsilon).sqrt()) / 2.;
}

const D: f64 = 0.001;

pub fn derivative_x(function: impl Fn(f64, f64) -> f64) -> impl Fn(f64, f64) -> f64 {
    move |x, y| (function(x + D, y) - function(x, y)) / D
}

pub fn derivative_y(function: impl Fn(f64, f64) -> f64) -> impl Fn(f64, f64) -> f64 {
    move |x, y| (function(x, y + D) - function(x, y)) / D
}
