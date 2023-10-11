use crate::util::{lin_map, smooth_maximum_unit};
use bevy::math::{DVec2, DVec3};
use itertools::{iproduct, Position};
use noise::utils::NoiseImage;
use noise::{NoiseFn, Perlin, Simplex};
use qoi::{decode_to_vec, encode_to_vec, Encoder};
use std::cmp::max;
use std::iter::Map;
use std::ops::Range;

/// Constants related to the resolution of the terrain signal
////// The amount of samples we take for the tile
pub const TILE_RESOLUTION: usize = 256usize;
////// The size of the tile in meters
pub const TILE_SIZE: f64 = 1000.;

/// Constants related to the amount of tiles that we're going to be rendering
pub const VIEW_RADIUS: i64 = 100;

/// Constants explicity related to the shape of the terrain
const HEIGHT_SCALING: f64 = 1.;
const MOUNTAIN_HEIGHT_OFFSET: f64 = 128. * 10.;
const MOUNTAIN_BIG_AMP: f64 = 90. * 10.;
const MOUNTAIN_SMALL_AMP: f64 = 5. * 10.;
const MOUNTAIN_MICRO_AMP: f64 = 20.;
pub const MAX_HEIGHT: f64 = HEIGHT_SCALING
    * (MOUNTAIN_HEIGHT_OFFSET + MOUNTAIN_BIG_AMP + MOUNTAIN_SMALL_AMP + MOUNTAIN_MICRO_AMP);
const FOOTHILL_START: f64 = 10. * TILE_SIZE;
const MOUNTAIN_OFFSET: f64 = MAX_HEIGHT * 1.1;

pub type HeightMap = Vec<[f64; TILE_RESOLUTION + 1]>;
pub type TerrainNormals = Vec<[DVec3; TILE_RESOLUTION + 1]>;

/// the tile's node position on the tile grid
pub type TilePosition = (i64, i64);

/// A function described by a simple lookup table
/// Only numeric currently
pub struct LookupTableFn {
    /// This value maps to `table[0]` when we sample
    pub domain_min: f64,
    /// This value maps to `table[TILE_RESOLUTION]` when we sample but
    /// since `table[TILE_RESOLUTION]` is not in the array obviously,
    /// the behavior is slightly more complex. This is just a good enough model
    pub domain_max: f64,

    /// The minimum value in the lookup table
    /// If the table is invertible, this will be `table[0]`
    pub range_min: f64,
    /// The max value in the lookup table
    /// If the table is invertible, this will be `table[1]`
    pub range_max: f64,

    pub table: [f64; TILE_RESOLUTION],
}

impl LookupTableFn {
    fn new(actual: impl Fn(f64) -> f64, min: f64, max: f64) -> Self {
        let mut table = [0.0; TILE_RESOLUTION];
        let mut range_min = actual(min);
        let mut range_max = actual(max);
        for i in 0..TILE_RESOLUTION {
            table[i] = actual(lin_map(0., TILE_RESOLUTION as f64, min, max, i as f64));
            if range_min > table[i] {
                range_min = table[i]
            }
            if range_max < table[i] {
                range_max = table[i]
            }
        }
        return LookupTableFn {
            domain_min: min,
            domain_max: max,
            range_min,
            range_max,
            table,
        };
    }

    fn sample(&self, x: f64) -> f64 {
        let idx = lin_map(
            self.domain_min,
            self.domain_max,
            0.,
            TILE_RESOLUTION as f64,
            x,
        )
        .round() as usize;
        return self.table[idx.max(TILE_RESOLUTION - 1).min(0)];
    }

    /// Integrate the function using the trapezoid method
    fn integrate(&self) -> f64 {
        let mut sum = 0.0;
        // The difference between sample points
        let d = lin_map(
            0.,
            TILE_RESOLUTION as f64,
            self.domain_min,
            self.domain_max,
            1.,
        ) - self.domain_min;
        for i in 1..TILE_RESOLUTION {
            sum += d * (self.table[i] - self.table[i - 1]) / 2. + self.table[i] * d;
        }
        return sum;
    }

    fn cdf_from_table(orig: &LookupTableFn) -> Self {
        let mut table = [0.0; TILE_RESOLUTION];
        let d = lin_map(
            0.,
            TILE_RESOLUTION as f64,
            orig.domain_min,
            orig.domain_max,
            1.,
        ) - orig.domain_min;
        for i in 0..TILE_RESOLUTION {
            table[i] = d * (orig.table[i] - orig.table[i - 1]) / 2.
                + orig.table[i] * d
                + orig.table[i] * d;
        }
        return LookupTableFn {
            domain_min: orig.domain_min,
            domain_max: orig.domain_max,
            range_min: table[0],
            range_max: table[TILE_RESOLUTION - 1],
            table,
        };
    }

    fn cdf(actual: impl Fn(f64) -> f64, min: f64, max: f64) -> Self {
        let mut table = [0.0; TILE_RESOLUTION];
        let d = lin_map(0., TILE_RESOLUTION as f64, min, max, 1.) - min;
        for i in 1..TILE_RESOLUTION {
            let cur = actual(i as f64 * d + min);
            let prev = actual((i - 1) as f64 * d + min);
            table[i] = d * (cur - prev) / 2. + cur * d;
        }
        return LookupTableFn {
            domain_min: min,
            domain_max: max,
            range_min: table[0],
            range_max: table[TILE_RESOLUTION - 1],
            table,
        };
    }

    /// Only works if the table is invertible
    /// Undefined behavior abound if used otherwise
    fn inverted(orig: &LookupTableFn) -> Self {
        let mut table = [0.0; TILE_RESOLUTION];
        for x in 0..TILE_RESOLUTION {
            let new_idx_lower = lin_map(
                orig.range_min,
                orig.range_max,
                0.,
                TILE_RESOLUTION as f64,
                orig.table[x - 1],
            )
            .round() as usize;
            let new_idx_upper = lin_map(
                orig.range_min,
                orig.range_max,
                0.,
                TILE_RESOLUTION as f64,
                orig.table[x],
            )
            .round() as usize;

            // fill
            for i in new_idx_lower..new_idx_upper {
                table[i.min(TILE_RESOLUTION - 1)] = lin_map(
                    0.,
                    TILE_RESOLUTION as f64,
                    orig.domain_min,
                    orig.domain_max,
                    i as f64,
                );
            }
            // force the ends
            table[new_idx_lower.min(TILE_RESOLUTION - 1)] = lin_map(
                0.,
                TILE_RESOLUTION as f64,
                orig.domain_min,
                orig.domain_max,
                x as f64 - 1.,
            );
            table[new_idx_upper.min(TILE_RESOLUTION - 1)] = lin_map(
                0.,
                TILE_RESOLUTION as f64,
                orig.domain_min,
                orig.domain_max,
                x as f64,
            );
        }
        return LookupTableFn {
            domain_min: orig.range_min,
            domain_max: orig.range_max,
            range_min: orig.domain_min,
            range_max: orig.domain_max,
            table,
        };
    }
}

#[derive(Clone)]
pub struct TileRelativePosition {
    /// The absolute coordinates on the tile plane
    absolute_position: TilePosition,
    /// The relative position from the player's current tile
    relative_position: TilePosition,
    /// The distance between the two above, in the same format as D from
    /// the TileMeshDescription
    distance: i64,
}

/// The list of LODS.
/// Each entry corresponds to a distance from the player's tile.
/// The keys correspond to square roots of squared distances.
/// Think of sqrt(D) where D = d^2.
/// Each keys corresponds to a D (which should be an int).
pub type TileMeshDescription = Map<i64, TileMeshLOD>;

pub struct TileMeshLOD {
    /// The relative detail
    position: TileRelativePosition,
}

/// The underlying "signal" of the terrain function
/// Calling it a signal because it looks like a wave
pub type TileSignal = (HeightMap, TerrainNormals);

pub struct TileLODDescription {
    /// The position of the LOD
    position: TileRelativePosition,
    /// The inverse cdf of the marginal
    marginal_inv_cdf: LookupTableFn,
    /// The inverse cdf's of the conditionals
    conditional_inv_cdfs: Vec<LookupTableFn>,
}

pub type TileLODS = Map<i64, TileLODDescription>;

pub struct Tile {
    /// The map of LOD's from
    lods: TileLODS,
    base: TileSignal,
}

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

        let freq1 = 0.02f64;
        let freq2 = 12.0 * freq1;
        let freq3 = freq2;

        let amp1 = MOUNTAIN_BIG_AMP;
        let amp2 = MOUNTAIN_SMALL_AMP;
        let amp3 = MOUNTAIN_MICRO_AMP;

        return self.mountain_noise_generators[0].get([x * freq1, z * freq1]) * amp1 / 2.
            + self.mountain_noise_generators[1].get([x * freq2, z * freq2]) * amp2 / 2.
            + self.mountain_noise_generators[2].get([x * freq3, z * freq3]) * amp3 / 2.
            + (amp1 + amp2 + amp3) / 2.
            + MOUNTAIN_HEIGHT_OFFSET;
    }

    fn get_valley_height_for(&self, x: f64, y: f64) -> f64 {
        let scale = 1.;
        let freq1 = 0.005;
        let freq2 = 0.03;

        let amp1 = 20.;
        let amp2 = 5.;
        return smooth_maximum_unit(
            self.valley_noise_generators[0].get([x * scale * freq1, y * scale * freq1]) * amp1 - 3.,
            self.valley_noise_generators[1].get([x * scale * freq2, y * scale * freq2]) * amp2,
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
        let d = 0.001;
        let terrain_height = self.get_height_for(x, y);
        let dx = self.get_height_for(x + d, y) - terrain_height;
        return dx / d;
    }

    fn get_y_partial_der_1st(&self, x: f64, y: f64) -> f64 {
        let d = 0.001;
        let terrain_height = self.get_height_for(x, y);
        let dy = self.get_height_for(x, y + d) - terrain_height;
        return dy / d;
    }

    pub fn get_normal(&self, x: f64, y: f64) -> DVec3 {
        return DVec3::new(
            self.get_x_partial_der_1st(x, y),
            1.,
            self.get_y_partial_der_1st(x, y),
        )
        .normalize();
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

/// Reads the already generated tile from the disk
pub fn get_LOD(tile_pos: TileRelativePosition) -> TileLODDescription {
    panic!("Not implemented");
}

// /// Computes the cdf of a newly generated tile
// pub fn generate_cdf(generator: &TerrainGenerator, tile_pos: TileRelativePosition) -> Cdf {
//     // get the left tile
//     let x = tile_pos.relative_position.0;
//     let y = tile_pos.relative_position.1;
//     let left_cdf = get_LOD(TileRelativePosition {
//         absolute_position: (
//             tile_pos.absolute_position.0 - 1,
//             tile_pos.absolute_position.1,
//         ),
//         relative_position: (x - 1, y),
//         distance: (x - 1) * (x - 1) + y * y,
//     })
//     .cdf;
//     let bottom_cdf = get_LOD(TileRelativePosition {
//         absolute_position: (
//             tile_pos.absolute_position.0,
//             tile_pos.absolute_position.1 - 1,
//         ),
//         relative_position: (x, y - 1),
//         distance: x * x + (y - 1) * (y - 1),
//     })
//     .cdf;
//     let corner_cdf = get_LOD(TileRelativePosition {
//         absolute_position: (
//             tile_pos.absolute_position.0 - 1,
//             tile_pos.absolute_position.1 - 1,
//         ),
//         relative_position: (x - 1, y - 1),
//         distance: (x - 1) * (x - 1) + (y - 1) * (y - 1),
//     })
//     .cdf;
//
//     let mut cdf: Cdf = vec![[0.; TILE_RESOLUTION]; TILE_RESOLUTION];
//
//     for (xpixel, ypixel) in iproduct!(0..TILE_RESOLUTION, 0..TILE_RESOLUTION) {
//         let xpos = (x as f64 + xpixel as f64 / TILE_RESOLUTION as f64) * TILE_SIZE;
//         let ypos = (y as f64 + ypixel as f64 / TILE_RESOLUTION as f64) * TILE_SIZE;
//
//         // this section we numerically calculate the sampling cdf
//         // the formula is c[n, m] = p[n, m] + c[n - 1, m] + c[n, m - 1] - c[n - 1, m - 1]
//         // where p is the probability dist and c is the cdf
//         let mut prevx;
//         let mut prevy;
//         let mut prevboth;
//         if xpixel == 0 && ypixel == 0 {
//             prevx = left_cdf[TILE_RESOLUTION - 1][ypixel];
//             prevy = bottom_cdf[xpixel][TILE_RESOLUTION - 1];
//             prevboth = corner_cdf[TILE_RESOLUTION - 1][TILE_RESOLUTION - 1];
//         } else if xpixel == 0 {
//             prevx = left_cdf[TILE_RESOLUTION - 1][ypixel];
//             prevy = cdf[0][ypixel - 1];
//             prevboth = left_cdf[TILE_RESOLUTION - 1][ypixel - 1];
//         } else if ypixel == 0 {
//             prevx = cdf[xpixel - 1][0];
//             prevy = bottom_cdf[xpixel][TILE_RESOLUTION - 1];
//             prevboth = bottom_cdf[xpixel - 1][TILE_RESOLUTION - 1];
//         } else {
//             prevx = cdf[xpixel - 1][ypixel];
//             prevy = cdf[xpixel][ypixel - 1];
//             prevboth = cdf[xpixel - 1][ypixel - 1];
//         }
//         cdf[xpixel][ypixel] = generator.get_sampling_prob(xpos, ypos) + prevx + prevy - prevboth;
//     }
//
//     return cdf;
// }

// pub fn get_inverse_cdf() ->

pub fn generate_tile_lod(
    generator: &TerrainGenerator,
    rel_pos: &TileRelativePosition,
) -> TileLODDescription {
    let x_off = TILE_SIZE * rel_pos.absolute_position.0 as f64;
    let y_off = TILE_SIZE * rel_pos.absolute_position.1 as f64;
    // create the marginal
    let mut marginal = LookupTableFn {
        domain_min: x_off,
        domain_max: x_off + TILE_SIZE,
        range_min: 0.0,
        range_max: 0.0,
        table: [0.0; TILE_RESOLUTION],
    };
    for i in 0..TILE_RESOLUTION {
        let x = lin_map(0., TILE_RESOLUTION as f64, 0., TILE_SIZE, i as f64);
        marginal.table[i] = LookupTableFn::new(
            |y| generator.get_sampling_prob(x + x_off, y + y_off, rel_pos.distance as f64),
            y_off,
            y_off + TILE_SIZE,
        )
        .integrate();
    }

    let marginal_inv_cdf = LookupTableFn::inverted(&LookupTableFn::cdf_from_table(&marginal));
    let mut conditional_inv_cdfs = vec![LookupTableFn {
        domain_min: 0.0,
        domain_max: 0.0,
        range_min: 0.0,
        range_max: 0.0,
        table: [0.0; TILE_RESOLUTION],
    }];
    for i in 0..TILE_RESOLUTION {
        let x = marginal_inv_cdf.table[i];
        conditional_inv_cdfs[i] = LookupTableFn::inverted(&LookupTableFn::cdf(
            |y| generator.get_sampling_prob(x + x_off, y + y_off, rel_pos.distance as f64),
            x_off,
            x_off + TILE_SIZE,
        ));
    }

    return TileLODDescription {
        position: rel_pos.clone(),
        marginal_inv_cdf,
        conditional_inv_cdfs,
    };
}

pub fn get_mesh_description(
    generator: &TerrainGenerator,
    player_pos: TilePosition,
) -> TileMeshDescription {
    /// We construct our sample points
    /// Then we look those up on the inverse CDF
    for x in -VIEW_RADIUS..VIEW_RADIUS {
        let bound = ((VIEW_RADIUS * VIEW_RADIUS - x * x) as f64).sqrt().ceil() as i64;
        for y in -bound..bound {
            let tile_pos = (player_pos.0 + x, player_pos.1 + y);
            let relative_pos = (x, y);
            let dist = x * x + y * y;
            let lod = get_LOD(TileRelativePosition {
                absolute_position: tile_pos,
                relative_position: relative_pos,
                distance: dist,
            });
        }
    }
    panic!("");
}

pub fn generate_tile(generator: &TerrainGenerator, tile_pos: TilePosition) -> TileSignal {
    let x = tile_pos.0;
    let y = tile_pos.1;
    let mut raw: HeightMap = vec![[0.; TILE_RESOLUTION + 1]; TILE_RESOLUTION + 1];
    let mut normal: TerrainNormals = vec![[DVec3::ZERO; TILE_RESOLUTION + 1]; TILE_RESOLUTION + 1];
    for (xpixel, ypixel) in iproduct!(0..TILE_RESOLUTION + 1, 0..TILE_RESOLUTION + 1) {
        let xpos = (x as f64 + xpixel as f64 / TILE_RESOLUTION as f64) * TILE_SIZE;
        let ypos = (y as f64 + ypixel as f64 / TILE_RESOLUTION as f64) * TILE_SIZE;
        raw[xpixel][ypixel] = generator.get_height_for(xpos, ypos);
        normal[xpixel][ypixel] = generator.get_normal(xpos, ypos);
    }
    return (raw, normal);
}
