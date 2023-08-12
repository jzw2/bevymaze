use crate::util::lin_map;
use itertools::iproduct;
use noise::utils::NoiseImage;
use noise::{NoiseFn, Perlin, Simplex};
use qoi::{decode_to_vec, encode_to_vec, Encoder};
use std::cmp::max;

const TILE_SIZE: usize = 1024usize;
const FOOTHILL_START: f64 = 1000.0;
const MOUNTAIN_OFFSET: f64 = 300.0;

pub type Tile = Vec<[f32; TILE_SIZE]>;

pub struct TerrainGenerator {
    mountain_noise_generators: [Perlin; 3],
    valley_noise_generators: [Simplex; 2],
}

impl TerrainGenerator {
    pub fn new() -> Self {
        Self {
            mountain_noise_generators: [Perlin::new(0), Perlin::new(1), Perlin::new(2)],
            valley_noise_generators: [Simplex::new(0), Simplex::new(1)],
        }
    }

    fn get_mountain_height_for(&self, x: f64, y: f64) -> f64 {
        let scale = 0.010f64;
        let x = x * scale;
        let z = y * scale;

        let freq1 = 0.2f64;
        let freq2 = 12.0 * 0.2;

        let amp1 = 90.0;
        let amp2 = 4.0;

        return self.mountain_noise_generators[0].get([x * freq1, z * freq1]) * amp1
            + self.mountain_noise_generators[2].get([x * freq2, z * freq2]) * amp2
            + (amp1 + amp2) / 2.
            + 128.;
    }

    fn get_valley_height_for(&self, x: f64, y: f64) -> f64 {
        let scale = 0.005f64;
        let freq = 0.5;
        let x = x * scale * freq;
        let y = y * scale * freq;

        let amp1 = 20.;
        let amp2 = 2.;
        return (self.valley_noise_generators[0].get([x, y]) * amp1)
            .max(self.valley_noise_generators[1].get([x, y]) * amp2 + amp2 / 2.);
    }

    fn get_height_for(&self, x: f64, y: f64) -> f64 {
        let foothill_start = FOOTHILL_START;
        let ramp_perc = lin_map(
            foothill_start,
            foothill_start + MOUNTAIN_OFFSET,
            0.0,
            1.0,
            x.abs(),
        );
        return if ramp_perc > 1.0 {
            self.get_mountain_height_for(x, y)
        } else if ramp_perc < 0.0 {
            self.get_valley_height_for(x, y)
        } else {
            let factor: f64 = TerrainGenerator::lin_ramp(ramp_perc);
            self.get_valley_height_for(x, y) * (1.0 - factor)
                + self.get_mountain_height_for(x, y) * factor
        };
    }

    fn lin_ramp(x: f64) -> f64 {
        return x.min(1.0).max(0.0);
    }
}

pub fn generate_tile(generator: &TerrainGenerator, x: i64, y: i64) -> Tile {
    let mut raw: Tile = vec![[0.; 1024]; 1024];
    for (xpixel, ypixel) in iproduct!(0..TILE_SIZE, 0..TILE_SIZE) {
        let xp = xpixel as f64;
        let yp = ypixel as f64;
        let terrain_height = generator.get_height_for(
            (TILE_SIZE as i64 * x) as f64 + xp,
            (TILE_SIZE as i64 * y) as f64 + yp,
        ) as f32;
        raw[xpixel][ypixel] = terrain_height;
    }
    return raw;
}
