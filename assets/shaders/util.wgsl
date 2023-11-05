#define_import_path bevymaze::util

/// Lin map
fn lin_map(a: f32, b: f32, c: f32, d: f32, x: f32) -> f32 {
    return (d - c) / (b - a) * (x - a) + c;
}

fn lin_map_vec(a: f32, b: f32, c: f32, d: f32, x: vec2<f32>) -> vec2<f32> {
    return (d - c) / (b - a) * (x - a) + c;
}