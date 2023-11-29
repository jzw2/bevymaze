use crate::terrain_gen::TILE_SIZE;
use crate::util::lin_map;
use bevy_easings::Lerp;
use byteorder::{NativeEndian, ReadBytesExt};
use std::io::BufReader;

// the construction of the tiles is such that the data zigzags across
// there's completely accidental relation that makes it so that the even bits
// are the x component in the matrix and odd bits are the y component
pub const TILE_DIM: usize = 1024;
pub const DATUM_COUNT: usize = TILE_DIM * TILE_DIM;
pub const TILE_DIM_LG: usize = 10;
#[derive(Clone)]
pub struct TerrainTile(Vec<u16>);

impl From<Vec<u8>> for TerrainTile {
    fn from(v: Vec<u8>) -> Self {
        let mut buf_reader = BufReader::new(v.as_slice());
        let mut buffer: Vec<u16> = Vec::with_capacity(DATUM_COUNT);
        unsafe {
            buffer.set_len(DATUM_COUNT);
        }
        buf_reader
            .read_u16_into::<NativeEndian>(&mut buffer[..])
            .expect("failed to read");
        return TerrainTile(buffer);
    }
}

impl From<Vec<u16>> for TerrainTile {
    fn from(value: Vec<u16>) -> Self {
        return TerrainTile(value);
    }
}

impl From<TerrainTile> for Vec<u16> {
    fn from(value: TerrainTile) -> Self {
        let TerrainTile(vec) = value;
        return vec;
    }
}

impl TerrainTile {
    /// We get the height of the terrain at a certain point based off of the LOD
    /// We use bilinear filtering
    /// We calculate the appropriate LOD based off the density for a particular section
    /// x and z are both in the range [0, TILE_SIZE]
    pub fn get_height(&self, x: f64, z: f64) -> f64 {
        let TerrainTile(data) = self;
        let dim = data.len();
        // the coordinates of the point relative to the chunk
        let x_offset = x / TILE_SIZE;
        let z_offset = z / TILE_SIZE;
        let x_idx = x_offset * dim as f64;
        let y_idx = z_offset * dim as f64;
        let x_frac = x_idx.fract();
        let y_frac = y_idx.fract();
        let min_x_idx = x_idx.floor() as usize;
        let max_x_idx = x_idx.ceil() as usize;
        let min_y_idx = y_idx.floor() as usize;
        let max_y_idx = y_idx.ceil() as usize;
        // lerped between the two "top" points (min_x, min_y) and (max_x, min_y)
        // l/h stand for low and high respectively
        let ylxl = uncompressed_height(data[coords_to_idx((min_x_idx, min_y_idx))]);
        let ylxh = uncompressed_height(data[coords_to_idx((max_x_idx, min_y_idx))]);
        let min_y_x_lerped = ylxl.lerp(&ylxh, &x_frac);
        // same as above but with max_y instead
        let yhxl = uncompressed_height(data[coords_to_idx((min_x_idx, max_y_idx))]);
        let yhxh = uncompressed_height(data[coords_to_idx((max_x_idx, max_y_idx))]);
        let max_y_x_lerped = yhxl.lerp(&yhxh, &x_frac);
        // finally lerp in the y direction
        return min_y_x_lerped.lerp(&max_y_x_lerped, &y_frac);
    }
}

pub fn compressed_height(h: f64) -> u16 {
    return lin_map(-500.0, 8500.0, 0.0, 0xFFFF as f64, h)
        .round()
        .min(0xFFFF as f64)
        .max(0.0) as u16;
}

pub fn uncompressed_height(h: u16) -> f64 {
    return lin_map(0.0, 0xFFFF as f64, -500.0, 8500.0, h as f64);
}

/// Copied from
/// https://stackoverflow.com/a/45695465/3210986
#[inline]
fn remove_odd_bits(mut x: u32) -> u32 {
    // x = 0b .a.b .c.d .e.f .g.h .i.j .k.l .m.n .o.p

    x = ((x & 0x44444444) >> 1) | ((x & 0x11111111) >> 0);
    // x = 0b ..ab ..cd ..ef ..gh ..ij ..kl ..mn ..op

    x = ((x & 0x30303030) >> 2) | ((x & 0x03030303) >> 0);
    // x = 0b .... abcd .... efgh .... ijkl .... mnop

    x = ((x & 0x0F000F00) >> 4) | ((x & 0x000F000F) >> 0);
    // x = 0b .... .... abcd efgh .... .... ijkl mnop

    x = ((x & 0x00FF0000) >> 8) | ((x & 0x000000FF) >> 0);
    // x = 0b .... .... .... .... abcd efgh ijkl mnop

    return x;
}

/// The following two algorithms are copied from
/// https://graphics.stanford.edu/~seander/bithacks.html
#[inline]
fn swap_bit_order(mut v: u32) -> u32 {
    // swap odd and even bits
    v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
    // swap consecutive pairs
    v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
    // swap nibbles ...
    v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
    // swap bytes
    v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
    // swap 2-byte long pairs
    return (v >> 16) | (v << 16);
}

#[inline]
fn interleave_bits(mut x: u32, mut y: u32) -> u32 {
    const B: [u32; 4] = [0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF];
    const S: [u32; 4] = [1, 2, 4, 8];

    x = (x | (x << S[3])) & B[3];
    x = (x | (x << S[2])) & B[2];
    x = (x | (x << S[1])) & B[1];
    x = (x | (x << S[0])) & B[0];

    y = (y | (y << S[3])) & B[3];
    y = (y | (y << S[2])) & B[2];
    y = (y | (y << S[1])) & B[1];
    y = (y | (y << S[0])) & B[0];

    return x | (y << 1);
}

/// As an example, our idx in a 8x8 grid may have the form ab cd ef
/// Our coords are then (fdb, eca)
/// This is removing the even (y) or odd (x) bits and then swapping the order
#[inline]
pub fn idx_to_coords(idx: usize) -> (usize, usize) {
    let idx = idx as u32;
    let x = swap_bit_order(remove_odd_bits(idx));
    // y uses even bits
    let y = swap_bit_order(remove_odd_bits(idx >> 1));
    // lastly correct for the offset since we have an erroneous 22 bits at the front
    return (
        (x >> (32 - TILE_DIM_LG)) as usize,
        (y >> (32 - TILE_DIM_LG)) as usize,
    );
}

#[inline]
pub fn coords_to_idx(coords: (usize, usize)) -> usize {
    let coords = (
        (coords.0 << (32 - TILE_DIM_LG)) as u32,
        (coords.1 << (32 - TILE_DIM_LG)) as u32,
    );
    let coords = (swap_bit_order(coords.0), swap_bit_order(coords.1));
    return interleave_bits(coords.0, coords.1) as usize;
}

/*
/// The lattice plane is the set of points that we transform to get the final view-normalized
/// terrain plane
/// We create our lattice plane within an ellipse
/// After it's transformed this just so happens to be a decent approximation for a rectangle
/// Calculating the number of vertices in the ellipse is analogous to computing the area
/// which happens to have a nice formula (a * b * pi = A, where a/b are the major/minor axes)
/// So we calculate the length of the axes and then we use the formula x^2/a^2 + y^2/b^2 = 1
/// to get the specific height/width of the ellipse at a certain position along one of the axes
pub fn create_lattice_plane() -> Vec<DVec3> {
    let mut verts: Vec<DVec3> = vec![];
    let x_bound = (X_VIEW_DIST_M*SCALE).asinh()/SCALE;
    let z_bound = (Z_VIEW_DIST_M*SCALE).asinh()/SCALE;
    let aspect_ratio = x_bound / z_bound;

    let z_0_sqr: f64 = 3. / 4.;
    let z_0 = z_0_sqr.sqrt();

    let x_verts = (z_0 * TERRAIN_VERTICES as f64 * aspect_ratio / PI)
        .sqrt()
        .ceil() as i32;
    let z_verts = (TERRAIN_VERTICES as f64 / (PI * x_verts as f64)).ceil() as i32;

    let total_verts = (PI * (x_verts * z_verts) as f64) as i32;
    println!(
        "x verts {} | z verts {} | total {} | maj axis {} (ex {}) | min axis {} (ex {})",
        x_verts,
        z_verts,
        total_verts,
        2. * z_verts as f64 / z_0,
        2. * z_bound,
        2. * x_verts as f64,
        2. * x_bound
    );

    for z_idx in -z_verts..z_verts {
        let z_pos = lin_map(
            -z_verts as f64,
            z_verts as f64,
            -z_bound,
            z_bound,
            z_idx as f64,
        );
        let x_bound_cur = (x_bound.powi(2) - aspect_ratio.powi(2) * z_pos.powi(2)).sqrt();
        let x_verts_above = (x_bound_cur / x_bound * x_verts as f64).ceil() as i32;

        for x_idx in -x_verts_above..x_verts_above {
            let x_pos = lin_map(
                -x_verts as f64,
                x_verts as f64,
                -x_bound,
                x_bound,
                x_idx as f64 + 0.5 * (z_idx % 2) as f64,
            );
            verts.push(DVec3::new(x_pos, 0., z_pos));
        }
    }

    return verts;
}


/// This function transforms to the lattice plane into our final terrain mesh
/// We also align our grid points to the nearest half meter to make fetching terrain data easier
/// Here we sample assuming the probability dist
/// p = 1/sqrt((ar)^2+1)
/// the cdf of p aka c is c = asinh(ar)/a
/// the inverse cdf is then sinh(ap)/a
/// This makes our math look quite simple
/// The most complicated bit is changing to polar coordinates to do the transformation
pub fn transform_lattice_positions(lattice: &mut Vec<DVec3>) {
    for lattice_pos in lattice {
        let pol = cart_to_polar((lattice_pos.x, lattice_pos.z));
        lattice_pos.x = (pol.0*SCALE).sinh()/SCALE * pol.1.cos();
        lattice_pos.z = (pol.0*SCALE).sinh()/SCALE * pol.1.sin();
        // round the transformed pos to the nearest grid pos (half of a meter)
        lattice_pos.x = (lattice_pos.x * 2.).round()/2.;
        lattice_pos.z = (lattice_pos.z * 2.).round()/2.;
    }
    // TODO: remove duplicates

    // for lattice_pos in lattice {
    //     if lattice_pos
    // }
}
*/
