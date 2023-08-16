use crate::util::lin_map;
use bevy::math::Vec3;
use itertools::iproduct;
use noise::utils::NoiseImage;
use noise::{NoiseFn, Perlin, Simplex};
use qoi::{decode_to_vec, encode_to_vec, Encoder};
use std::cmp::max;

const TILE_RESOLUTION: usize = 128usize;
pub const TILE_SIZE: f64 = 100.;
const FOOTHILL_START: f64 = 20.*TILE_SIZE;
const MOUNTAIN_OFFSET: f64 = 300.;
const HEIGHT_SCALING: f64 = 1.;
const MOUNTAIN_HEIGHT_OFFSET: f64 = 128.;
const MOUNTAIN_BIG_AMP: f64 = 90.;
const MOUNTAIN_SMALL_AMP: f64 = 5.;
pub const MAX_HEIGHT: f64 =
    HEIGHT_SCALING * (MOUNTAIN_HEIGHT_OFFSET + MOUNTAIN_BIG_AMP + MOUNTAIN_SMALL_AMP);

pub type HeightMap = Vec<Vec<f32>>;
pub type TerrainNormals = Vec<Vec<Vec3>>;
pub type Tile = (HeightMap, TerrainNormals);

pub struct TerrainGenerator {
    mountain_noise_generators: [Perlin; 3],
    valley_noise_generators: [Perlin; 2],
}

impl TerrainGenerator {
    pub fn new() -> Self {
        Self {
            mountain_noise_generators: [Perlin::new(0), Perlin::new(1), Perlin::new(2)],
            valley_noise_generators: [Perlin::new(3), Perlin::new(4)],
        }
    }

    fn get_mountain_height_for(&self, x: f64, y: f64) -> f64 {
        let scale = 0.025f64;
        let x = x * scale;
        let z = y * scale;

        let freq1 = 0.2f64;
        let freq2 = 12.0 * 0.2;

        let amp1 = MOUNTAIN_BIG_AMP;
        let amp2 = MOUNTAIN_SMALL_AMP;

        return self.mountain_noise_generators[0].get([x * freq1, z * freq1]) * amp1 / 2.
            + self.mountain_noise_generators[2].get([x * freq2, z * freq2]) * amp2 / 2.
            + (amp1 + amp2) / 2.
            + MOUNTAIN_HEIGHT_OFFSET;
    }

    fn get_valley_height_for(&self, x: f64, y: f64) -> f64 {
        let scale = 0.5;
        let freq1 = 0.015;
        let freq2 = 0.03;

        let amp1 = 10.;
        let amp2 = 1.;
        return (self.valley_noise_generators[0].get([x * scale * freq1, y * scale * freq1])
            * amp1 - 3.)
            .max(
                self.valley_noise_generators[1].get([x * scale * freq2, y * scale * freq2]) * amp2,
            );
    }

    fn get_height_for_unscaled(&self, x: f64, y: f64) -> f64 {
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

    fn get_height_for(&self, x: f64, y: f64) -> f64 {
        return self.get_height_for_unscaled(x, y) * HEIGHT_SCALING;
    }

    fn lin_ramp(x: f64) -> f64 {
        return x.min(1.0).max(0.0);
    }
}

pub fn generate_tile(generator: &TerrainGenerator, x: i64, y: i64) -> Tile {
    let mut raw: HeightMap = vec![vec![0.; TILE_RESOLUTION]; TILE_RESOLUTION];
    let mut normal: TerrainNormals = vec![vec![Vec3::ZERO; TILE_RESOLUTION]; TILE_RESOLUTION];
    for (xpixel, ypixel) in iproduct!(0..TILE_RESOLUTION, 0..TILE_RESOLUTION) {
        let xpos = (x as f64 + xpixel as f64 / (TILE_RESOLUTION - 1) as f64) * TILE_SIZE;
        let ypos = (y as f64 + ypixel as f64 / (TILE_RESOLUTION - 1) as f64) * TILE_SIZE;
        let terrain_height = generator.get_height_for(xpos, ypos);
        let d = 0.001;
        let dx = generator.get_height_for(xpos + d as f64, ypos) - terrain_height;
        let dz = generator.get_height_for(xpos, ypos + d as f64) - terrain_height;
        raw[xpixel][ypixel] = terrain_height as f32;
        normal[xpixel][ypixel] = Vec3::new(0., dz as f32, d)
            .cross(Vec3::new(d, dx as f32, 0.))
            .normalize();
    }

    // now on the borders of the tiles, make sure we're using the height on the very edge of the tile
    // for smooth tile transitions
    // for i in 0..TILE_RESOLUTION {
    //     // this gets the points of the left edge on the tile to the right
    //     raw[TILE_RESOLUTION - 1][i] = generator.get_height_for(
    //         (TILE_RESOLUTION as i64 * (x + 1)) as f64,
    //         (TILE_RESOLUTION as i64 * y) as f64 + i as f64,
    //     ) as f32;
    //     raw[i][TILE_RESOLUTION - 1] = generator.get_height_for(
    //         (TILE_RESOLUTION as i64 * x) as f64 + i as f64,
    //         (TILE_RESOLUTION as i64 * (y + 1)) as f64,
    //     ) as f32;
    // }
    return (raw, normal);
}
