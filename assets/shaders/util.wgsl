#define_import_path bevymaze::util

/// Lin map
fn lin_map(a: f32, b: f32, c: f32, d: f32, x: f32) -> f32 {
    return (d - c) / (b - a) * (x - a) + c;
}

fn lin_map_vec(a: f32, b: f32, c: f32, d: f32, x: vec2<f32>) -> vec2<f32> {
    return (d - c) / (b - a) * (x - a) + c;
}

fn hash(in: u32) -> f32 {
    // integer hash copied from Hugo Elias
    var n = in;
    n = (n << 13u) ^ n;
    n = n * (n * n * 15731u + 789221u) + 1376312589u;
    return f32(n & u32(0x7fffffffu)) / f32(0x7fffffffu);
}

/// Copied from https://www.shadertoy.com/view/ttc3zr
fn murmurHash11(inp: u32) -> u32 {
    var src = inp;
    let M: u32 = 0x5bd1e995u;
    var h: u32 = 1190494759u;
    src *= M; src ^= src >> 24u; src *= M;
    h *= M; h ^= src;
    h ^= h >> 13u; h *= M; h ^= h >> 15u;
    return h;
}

// 1 output, 1 input
fn hash11(src: f32) -> f32 {
    let h = murmurHash11(bitcast<u32>(src));
    return bitcast<f32>(h & 0x007fffffu | 0x3f800000u) - 1.0;
}

fn murmurHash12(inp: vec2<u32>) -> u32 {
    var src = inp;
    let M: u32 = 0x5bd1e995u;
    var h: u32 = 1190494759u;
    src *= M; src ^= src >> 24u; src *= M;
    h *= M; h ^= src.x; h *= M; h ^= src.y;
    h ^= h >> 13u; h *= M; h ^= h >> 15u;
    return h;
}

// 1 output, 2 inputs
fn hash12(src: vec2<f32>) -> f32 {
    let h = murmurHash12(bitcast<u32>(src));
    return bitcast<f32>(h & 0x007fffffu | 0x3f800000u) - 1.0;
}

fn positive_rem(n: f32, m: f32) -> f32 {
  return ((n % m) + m) % m;
}