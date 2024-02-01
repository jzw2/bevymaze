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

#[inline]
pub fn barycentric(point: &[f64; 2], tri: &[[f64; 2]; 3]) -> (f64, f64) {
    let p0 = tri[0];
    let p1 = tri[1];
    let p2 = tri[2];
    let area =
        0.5 * (-p1[1] * p2[0] + p0[1] * (-p1[0] + p2[0]) + p0[0] * (p1[1] - p2[1]) + p1[0] * p2[1]);
    let s = 1. / (2. * area)
        * (p0[1] * p2[0] - p0[0] * p2[1] + (p2[1] - p0[1]) * point[0] + (p0[0] - p2[0]) * point[1]);
    let t = 1. / (2. * area)
        * (p0[0] * p1[1] - p0[1] * p1[0] + (p0[1] - p1[1]) * p0[0] + (p1[0] - p0[0]) * point[1]);
    return (s, t);
}

fn dot(p1: &[f32; 2], p2: &[f32; 2]) -> f32 {
    return p1[0] * p2[0] + p1[1] * p2[1];
}

fn sub(p1: &[f32; 2], p2: &[f32; 2]) -> [f32; 2] {
    return [p1[0] - p2[0], p1[1] - p2[1]];
}

#[inline]
pub fn barycentric32(point: &[f32; 2], tri: &[[f32; 2]; 3]) -> (f32, f32) {
    let p = point;
    let [a, b, c] = tri;
    let v0 = &sub(b, a);
    let v1 = &sub(c, a);
    let v2 = &sub(p, a);
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let denom = d00 * d11 - d01 * d01;
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    return (v, w);
}

#[inline(always)]
/// Copied and adapted from
/// https://observablehq.com/@mootari/delaunay-findtriangle
/// Returns the orientation of three points A, B and C:
///   -1 = counterclockwise
///    0 = collinear
///    1 = clockwise
/// More on the topic: http://www.dcs.gla.ac.uk/~pat/52233/slides/Geometry1x1.pdf
pub fn orientation(a: &[f32; 2], b: &[f32; 2], c: &[f32; 2]) -> f32 {
    // Determinant of vectors of the line segments AB and BC:
    // [ cx - bx ][ bx - ax ]
    // [ cy - by ][ by - ay ]
    let [ax, ay] = a;
    let [bx, by] = b;
    let [cx, cy] = c;
    return ((cx - bx) * (by - ay) - (cy - by) * (bx - ax)).signum();
}

/// Copied and adapted from https://stackoverflow.com/a/2049593/3210986
#[inline(always)]
pub fn point_in_triangle(p: &[f32; 2], tri: &[[f32; 2]; 3]) -> bool {
    let [v1, v2, v3] = tri;
    let d1 = orientation(p, v1, v2);
    let d2 = orientation(p, v2, v3);
    let d3 = orientation(p, v3, v1);

    let has_neg = (d1 < 0.) || (d2 < 0.) || (d3 < 0.);
    let has_pos = (d1 > 0.) || (d2 > 0.) || (d3 > 0.);

    return !(has_neg && has_pos);
}

/// Determine if the point is in the triangle using barycentric coordinates
#[inline]
pub fn point_in_triangle_bary(point: &[f64; 2], tri: &[[f64; 2]; 3]) -> bool {
    let (s, t) = barycentric(point, tri);
    return s >= 0. && t >= 0. && 1. - s - t >= 0.;
}
