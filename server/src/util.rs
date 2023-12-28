use std::f64::consts::PI;

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

/// Convert a polar coordinate to it's cartesian counterpart
pub fn polar_to_cart(p: (f64, f64)) -> (f64, f64) {
    return (p.0 * p.1.cos(), p.0 * p.1.sin());
}

/// Get the distance between two points
pub fn dist(p1: (f64, f64), p2: (f64, f64)) -> f64 {
    return origin_dist((p1.0 - p2.0, p1.1 - p2.1));
}

/// Get the distance between a point and the origin
pub fn origin_dist(p: (f64, f64)) -> f64 {
    return (p.0 * p.0 + p.1 * p.1).sqrt();
}

/// Get the angle that the segment between the origin and p makes with the x-axis
/// Return value is in range from [0, 2pi)
pub fn polar_angle(p: (f64, f64)) -> f64 {
    return (p.1.atan2(p.0) + (2.0 * PI)).rem_euclid(2.0 * PI);
}

/// Convert a cartesian coordinate to polar
pub fn cart_to_polar(p: (f64, f64)) -> (f64, f64) {
    return (origin_dist(p), polar_angle(p));
}

pub fn barycentric(point: &[f64; 2], tri: &[[f64; 2]; 3]) -> (f64, f64) {
    let p0 = tri[0];
    let p1 = tri[1];
    let p2 = tri[2];
    let area = 0.5
        * (-p1[1] * p2[0]
            + p0[1] * (-p1[0] + p2[0])
            + p0[0] * (p1[1] - p2[1])
            + p1[0] * p2[1]);
    let s = 1. / (2. * area)
        * (p0[1] * p2[0] - p0[0] * p2[1]
            + (p2[1] - p0[1]) * point[0]
            + (p0[0] - p2[0]) * point[1]);
    let t = 1. / (2. * area)
        * (p0[0] * p1[1] - p0[1] * p1[0]
            + (p0[1] - p1[1]) * p0[0]
            + (p1[0] - p0[0]) * point[1]);
    return (s, t);
}

pub fn intersects_tri(point: &[f64; 2], tri: &[[f64; 2]; 3]) -> bool {
    let (s, t) = barycentric(point, tri);
    return s >= 0. && t >= 0. && 1. - s - t >= 0.;
}
