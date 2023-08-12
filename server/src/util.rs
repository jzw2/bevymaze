pub fn lin_map(a: f64, b: f64, c: f64, d: f64, x: f64) -> f64 {
    return (d - c) / (b - a) * (x - a) + c;
}

pub fn lin_map32(a: f32, b: f32, c: f32, d: f32, x: f32) -> f32 {
    return (d - c) / (b - a) * (x - a) + c;
}