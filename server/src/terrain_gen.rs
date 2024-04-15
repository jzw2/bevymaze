use crate::terrain_data::{compressed_height, idx_to_coords, DATUM_COUNT, TILE_DIM};
use crate::util::{lin_map, smooth_maximum_unit};
use bevy::math::{DVec2, DVec3};
use libnoise::prelude::*;
/// Constants related to the resolution of the terrain signal
////// The amount of samples we take for the tile
pub const TILE_RESOLUTION: usize = 256usize;
////// The size of the tile in meters
pub const TILE_SIZE: f64 = 1024.;

/// Constants related to the amount of tiles that we're going to be rendering
pub const VIEW_RADIUS: i64 = 100;

/// Constants explicity related to the shape of the terrain
const HEIGHT_SCALING: f64 = 1.;
const MOUNTAIN_HEIGHT_OFFSET: f64 = 128. * 10.;
const MOUNTAIN_MASSIVE_AMP: f64 = 100.;
const MOUNTAIN_BIG_AMP: f64 = 90. * 10.;
const MOUNTAIN_SMALL_AMP: f64 = 5. * 10.;
const MOUNTAIN_MICRO_AMP: f64 = 20.;
pub const MAX_HEIGHT: f64 = HEIGHT_SCALING
    * (MOUNTAIN_HEIGHT_OFFSET
        + MOUNTAIN_BIG_AMP
        + MOUNTAIN_SMALL_AMP
        + MOUNTAIN_MICRO_AMP
        + MOUNTAIN_MASSIVE_AMP);
pub const FOOTHILL_START: f64 = 10. * TILE_SIZE;
const MOUNTAIN_OFFSET: f64 = MAX_HEIGHT * 1.1;

const D: f64 = 0.001;

pub type HeightMap = Vec<[f64; TILE_RESOLUTION + 1]>;
pub type TerrainNormals = Vec<[DVec3; TILE_RESOLUTION + 1]>;

/// the tile's node position on the tile grid
pub type TilePosition = (i64, i64);

pub struct TerrainGenerator {
    mountain_noise_generators: [Simplex<2>; 4],
    valley_noise_generators: [Simplex<2>; 2],
}

impl TerrainGenerator {
    pub fn new() -> Self {
        Self {
            mountain_noise_generators: [
                Source::simplex(10),
                Source::simplex(11),
                Source::simplex(12),
                Source::simplex(13),
            ],
            valley_noise_generators: [Source::simplex(13), Source::simplex(14)],
        }
    }

    fn get_mountain_height_for(&self, x: f64, y: f64) -> f64 {
        let scale = 0.025f64;
        let x = x * scale;
        let z = y * scale;

        let freq0 = 0.001f64;
        let freq1 = 0.02f64;
        let freq2 = 12.0 * freq1;
        let freq3 = freq2;

        let amp0 = MOUNTAIN_MASSIVE_AMP;
        let amp1 = MOUNTAIN_BIG_AMP;
        let amp2 = MOUNTAIN_SMALL_AMP;
        let amp3 = MOUNTAIN_MICRO_AMP;

        return self.mountain_noise_generators[0].sample([x * freq0, z * freq0]) * amp0 / 2.
            + self.mountain_noise_generators[1].sample([x * freq1, z * freq1]) * amp1 / 2.
            + self.mountain_noise_generators[2].sample([x * freq2, z * freq2]) * amp2 / 2.
            + self.mountain_noise_generators[3].sample([x * freq3, z * freq3]) * amp3 / 2.
            + (amp1 + amp2 + amp3) / 2.
            + MOUNTAIN_HEIGHT_OFFSET;
    }

    fn get_valley_height_for(&self, x: f64, y: f64) -> f64 {
        let scale = 1.;
        let freq1 = 0.0025;
        let freq2 = 0.003;

        let amp1 = 20.;
        let amp2 = 5.;
        return smooth_maximum_unit(
            self.valley_noise_generators[0].sample([x * scale * freq1, y * scale * freq1]) * amp1
                - 3.,
            self.valley_noise_generators[1].sample([x * scale * freq2, y * scale * freq2]) * amp2,
            2.,
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

    pub fn get_height_for(&self, x: f64, y: f64) -> f64 {
        return self.get_height_for_unscaled(x, y) * HEIGHT_SCALING;
    }

    fn get_x_partial_der_1st(&self, x: f64, y: f64) -> f64 {
        let terrain_height = self.get_height_for(x, y);
        let dx = self.get_height_for(x + D, y) - terrain_height;
        return dx / D;
    }

    fn get_y_partial_der_1st(&self, x: f64, y: f64) -> f64 {
        let terrain_height = self.get_height_for(x, y);
        let dy = self.get_height_for(x, y + D) - terrain_height;
        return dy / D;
    }

    pub fn get_normal(&self, x: f64, y: f64) -> DVec3 {
        let terrain_height = self.get_height_for(x, y);

        let dx = self.get_height_for(x + D, y) - terrain_height;

        let dy = self.get_height_for(x, y + D) - terrain_height;

        return DVec3::new(-dx / D, 1.0, -dy / D).normalize();
    }

    pub fn get_gradient(&self, x: f64, y: f64) -> DVec2 {
        let terrain_height = self.get_height_for(x, y);

        let dx = self.get_height_for(x + D, y) - terrain_height;

        let dy = self.get_height_for(x, y + D) - terrain_height;

        return DVec2::new(dx / D, dy / D);
    }

    /// The formula here is simply
    /// exp(-|H_t|^2) + eps
    /// where H is the hessian and eps is a normalizing term that makes it so
    /// the probability of sampling a given point is never too close to 0
    fn get_sampling_prob(&self, x: f64, y: f64, dist_sqr: f64) -> f64 {
        // If you use the limit definition for the partial derivatives, you can
        // see better what I'm doing here. It's a lot of clerical stuff so I'm omitting it
        // from the comments. Sorry
        let d = 0.001;
        // useful intermediaries
        let f = self.get_height_for(x, y);
        let fxd = self.get_height_for(x + d, y);
        let fyd = self.get_height_for(x, y + d);
        let d2 = d * d;
        // actual derivatives
        let fxx = (self.get_height_for(x + 2. * d, y) - 2. * fxd + f) / d2;
        let fyy = (self.get_height_for(x, y + 2. * d) - 2. * fyd + f) / d2;
        let fxy = (self.get_height_for(x + d, y + d) - fxd - fyd + f) / d2;

        let hessian_det = fxx * fyy - fxy * fxy;

        return 1. / (dist_sqr + 1.).sqrt() * (-hessian_det * hessian_det).exp() + 0.000000001;
    }

    fn lin_ramp(x: f64) -> f64 {
        return x.min(1.0).max(0.0);
    }
}

/// Generates the raw u16 data for a given chunk
/// TODO: figure out way to merge this with the code in handle connection
/// TODO: main issue is that I don't want to run the loop twice, but I want this to be
/// TODO: generic *shrug*
pub fn generate_data(buffer: &mut Vec<u16>, chunk: (i32, i32)) {
    let generator = TerrainGenerator::new();

    for i in 0..DATUM_COUNT {
        let (x, y) = idx_to_coords(i);
        let x_world_pos =
            lin_map(0., TILE_DIM as f64, 0., TILE_SIZE, x as f64) + TILE_SIZE * chunk.0 as f64;
        let z_world_pos =
            lin_map(0., TILE_DIM as f64, 0., TILE_SIZE, y as f64) + TILE_SIZE * chunk.1 as f64;
        buffer[i] = compressed_height(generator.get_height_for(x_world_pos, z_world_pos));
    }
}
