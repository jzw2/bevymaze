use crate::render::{
    quad_cc_indices, quad_cc_indices_off, quad_cw_indices_off, tri_cc_indices_off,
    tri_cw_indices_off, CompleteVertices,
};
use bevy::math::Vec3;
use bevy::prelude::Mesh;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use itertools::iproduct;
use lazy_static::lazy_static;
use server::terrain_gen::Tile;
use server::util::{lin_map, lin_map32};
use std::os::unix::raw::time_t;

pub const TILE_WORLD_SIZE: f32 = 20.;
const BASE_VERTICES: f64 = 16.;
const FREE_SUBDIVISIONS: f64 = 5.;
const HEIGHT_SCALING: f32 = 0.05;
lazy_static! {
    pub static ref BASE_SUBDIVISIONS: f64 = BASE_VERTICES.sqrt();
}

type TileOffset = (i64, i64);

fn get_tile_dist(t1_off: TileOffset, t2_off: TileOffset) -> f64 {
    let d = (t1_off.0 - t2_off.0, t1_off.1 - t1_off.1);
    return ((d.0 * d.0 + d.1 * d.1) as f64).sqrt();
}

fn get_side_subdivs(d: f64) -> f32 {
    return ((*BASE_SUBDIVISIONS / (d + 1.)) + FREE_SUBDIVISIONS).round() as f32;
}

/// Gets the height of the tile for a tile position
/// `xr` and `yr` are in the range `[0, 1]`
pub fn get_tile_height(xr: f32, yr: f32, tile: &Tile) -> f32 {
    let dims = tile.len() as f32 - 1.;
    return tile[(xr * dims).round() as usize][(yr * dims).round() as usize] * HEIGHT_SCALING;
}

pub fn get_tile_normal(xr: f32, yr: f32, tile: &Tile) -> [f32; 3] {
    let dims = tile.len() as f32 - 1.;
    let center = ((xr * dims).round() as usize, (yr * dims).round() as usize);
    let mut dx = 0.;
    if center.0 < tile.len() - 1 && center.0 > 0 {
        let dxm1 = tile[center.0][center.1] - tile[center.0 - 1][center.1];
        let dxp1 = tile[center.0 + 1][center.1] - tile[center.0][center.1];
        dx = (dxp1 + dxm1) / 2.;
    } else if center.0 > 0 {
        dx = tile[center.0][center.1] - tile[center.0 - 1][center.1];
    } else {
        dx = tile[center.0 + 1][center.1] - tile[center.0][center.1];
    }

    let mut dz = 0.;
    if center.1 < tile.len() - 1 && center.1 > 0 {
        let dzm1 = tile[center.0][center.1] - tile[center.0][center.1 - 1];
        let dzp1 = tile[center.0][center.1 + 1] - tile[center.0][center.1];
        dz = (dzp1 + dzm1) / 2.;
    } else if center.1 > 0 {
        dz = tile[center.0][center.1] - tile[center.0][center.1 - 1];
    } else {
        dz = tile[center.0][center.1 + 1] - tile[center.0][center.1];
    }

    return Vec3::new(0., dz, 1.)
        .cross(Vec3::new(1., dx, 0.))
        .to_array();
}

/// Gets the mesh for a tile
/// Fixes cracks between tiles by being smart about geometry
pub fn get_tile_mesh(tile_offset: TileOffset, tile: &Tile) -> Mesh {
    let subdivs = get_side_subdivs(get_tile_dist(tile_offset, (0, 0)));
    // let top_subdivs = get_side_subdivs(get_tile_dist(tile_offset, (0, -1)));
    let bottom_subdivs = get_side_subdivs(get_tile_dist(tile_offset, (0, 1)));
    // let left_subdivs = get_side_subdivs(get_tile_dist(tile_offset, (-1, 0)));
    let right_subdivs = get_side_subdivs(get_tile_dist(tile_offset, (1, 0)));

    let mut vertices: CompleteVertices = vec![];
    let mut indices: Vec<u32> = vec![];
    let mut cur_idx_set: u32 = 0;
    // the inner triangles are the simplest to create
    for xi in 0..(subdivs as u32) - 1 {
        for yi in 0..(subdivs as u32) - 1 {
            let xi0 = xi as f32 / subdivs;
            let xi1 = (xi + 1) as f32 / subdivs;
            let yi0 = yi as f32 / subdivs;
            let yi1 = (yi + 1) as f32 / subdivs;
            let v1 = [
                xi0 * TILE_WORLD_SIZE,
                get_tile_height(xi0, yi0, tile),
                yi0 * TILE_WORLD_SIZE,
            ];
            let v2 = [
                xi1 * TILE_WORLD_SIZE,
                get_tile_height(xi1, yi0, tile),
                yi0 * TILE_WORLD_SIZE,
            ];
            let v3 = [
                xi0 * TILE_WORLD_SIZE,
                get_tile_height(xi0, yi1, tile),
                yi1 * TILE_WORLD_SIZE,
            ];
            let v4 = [
                xi1 * TILE_WORLD_SIZE,
                get_tile_height(xi1, yi1, tile),
                yi1 * TILE_WORLD_SIZE,
            ];
            vertices.append(&mut vec![
                (v1, get_tile_normal(xi0, yi0, tile), [0.0, 0.]),
                (v2, get_tile_normal(xi1, yi0, tile), [1.0, 0.]),
                (v3, get_tile_normal(xi0, yi1, tile), [0.0, 1.]),
                (v4, get_tile_normal(xi1, yi1, tile), [1.0, 1.]),
            ]);
            indices.append(&mut quad_cc_indices(cur_idx_set));
            cur_idx_set += 1;
        }
    }

    // now do the sides
    // start with the bottom
    // we iterate the more-subdivided side
    // we check if the verts involved create a triangle or a quad
    // and break up the quads if necessary
    // first we do the quads

    // also since we're mixing quads and triangles we're going to keep track of the idx offset ourselves
    let mut idx_off = cur_idx_set * 4;

    let mut y0r = (subdivs - 1.) / subdivs;
    let mut y1r = 1.;
    let mut greater_subdivs = subdivs;
    let mut lesser_subdivs = bottom_subdivs;
    if subdivs <= bottom_subdivs {
        y0r = 1.;
        y1r = (subdivs - 1.) / subdivs;
        greater_subdivs = bottom_subdivs;
        lesser_subdivs = subdivs;
    }
    for xi in 0..greater_subdivs as u32 - 1 {
        let x0r = xi as f32 / greater_subdivs;
        let x1r = (xi + 1) as f32 / greater_subdivs;
        let c0r = (xi as f32 * lesser_subdivs / greater_subdivs).round() / lesser_subdivs;
        let c1r = ((xi + 1) as f32 * lesser_subdivs / greater_subdivs).round() / lesser_subdivs;
        let v1 = [
            x0r * TILE_WORLD_SIZE,
            get_tile_height(x0r, y0r, tile),
            y0r * TILE_WORLD_SIZE,
        ];
        let v2 = [
            x1r * TILE_WORLD_SIZE,
            get_tile_height(x1r, y0r, tile),
            y0r * TILE_WORLD_SIZE,
        ];
        let v3 = [
            c0r * TILE_WORLD_SIZE,
            get_tile_height(c0r, y1r, tile),
            y1r * TILE_WORLD_SIZE,
        ];
        if c0r != c1r {
            let v4 = [
                c1r * TILE_WORLD_SIZE,
                get_tile_height(c1r, y1r, tile),
                y1r * TILE_WORLD_SIZE,
            ];
            vertices.append(&mut vec![
                (v1, get_tile_normal(x0r, y0r, tile), [0.0, 0.]),
                (v2, get_tile_normal(x1r, y0r, tile), [1.0, 0.]),
                (v3, get_tile_normal(c0r, y1r, tile), [0.0, 1.]),
                (v4, get_tile_normal(c1r, y1r, tile), [1.0, 1.]),
            ]);
            indices.append(&mut quad_cw_indices_off(idx_off));
            idx_off += 4;
        } else {
            vertices.append(&mut vec![
                (v1, get_tile_normal(x0r, y0r, tile), [0.0, 0.]),
                (v2, get_tile_normal(x1r, y0r, tile), [1.0, 0.]),
                (v3, get_tile_normal(c0r, y1r, tile), [0.0, 1.]),
            ]);
            indices.append(&mut tri_cw_indices_off(idx_off));
            idx_off += 3;
        }
    }

    let mut x0r = (subdivs - 1.) / subdivs;
    let mut x1r = 1.;
    let mut greater_subdivs = subdivs;
    let mut lesser_subdivs = right_subdivs;
    if subdivs <= right_subdivs {
        x0r = 1.;
        x1r = (subdivs - 1.) / subdivs;
        greater_subdivs = right_subdivs;
        lesser_subdivs = subdivs;
    }
    for yi in 0..greater_subdivs as u32 - 1 {
        let y0r = yi as f32 / greater_subdivs;
        let y1r = (yi + 1) as f32 / greater_subdivs;
        let c0r = (yi as f32 * lesser_subdivs / greater_subdivs).round() / lesser_subdivs;
        let c1r = ((yi + 1) as f32 * lesser_subdivs / greater_subdivs).round() / lesser_subdivs;
        let v1 = [
            x0r * TILE_WORLD_SIZE,
            get_tile_height(x0r, y0r, tile),
            y0r * TILE_WORLD_SIZE,
        ];
        let v2 = [
            x1r * TILE_WORLD_SIZE,
            get_tile_height(x1r, c0r, tile),
            c0r * TILE_WORLD_SIZE,
        ];
        let v3 = [
            x0r * TILE_WORLD_SIZE,
            get_tile_height(x0r, y1r, tile),
            y1r * TILE_WORLD_SIZE,
        ];
        if c0r != c1r {
            let v4 = [
                x1r * TILE_WORLD_SIZE,
                get_tile_height(x1r, c1r, tile),
                c1r * TILE_WORLD_SIZE,
            ];
            vertices.append(&mut vec![
                (v1, get_tile_normal(x0r, y0r, tile), [0.0, 0.]),
                (v2, get_tile_normal(x1r, c0r, tile), [1.0, 0.]),
                (v3, get_tile_normal(x0r, y1r, tile), [0.0, 1.]),
                (v4, get_tile_normal(x1r, c1r, tile), [1.0, 1.]),
            ]);
            indices.append(&mut quad_cw_indices_off(idx_off));
            idx_off += 4;
        } else {
            vertices.append(&mut vec![
                (v1, get_tile_normal(x0r, y0r, tile), [0.0, 0.]),
                (v2, get_tile_normal(x1r, c0r, tile), [1.0, 0.]),
                (v3, get_tile_normal(x0r, y1r, tile), [0.0, 1.]),
            ]);
            indices.append(&mut tri_cw_indices_off(idx_off));
            idx_off += 3;
        }
    }

    // lastly we do the bottom left corner
    let v1r = (subdivs - 1.) / subdivs;
    let v1 = [
        v1r * TILE_WORLD_SIZE,
        get_tile_height(v1r, v1r, tile),
        v1r * TILE_WORLD_SIZE,
    ];
    let left_r = (right_subdivs - 1.) / right_subdivs;
    let v2 = [
        TILE_WORLD_SIZE,
        get_tile_height(1., left_r, tile),
        left_r * TILE_WORLD_SIZE,
    ];
    let bottom_r = (bottom_subdivs - 1.) / subdivs;
    let v3 = [
        bottom_r * TILE_WORLD_SIZE,
        get_tile_height(bottom_r, 1., tile),
        TILE_WORLD_SIZE,
    ];
    let v4 = [
        TILE_WORLD_SIZE,
        get_tile_height(1., 1., tile),
        TILE_WORLD_SIZE,
    ];
    vertices.append(&mut vec![
        (v1, get_tile_normal(v1r, v1r, tile), [0.0, 0.]),
        (v2, get_tile_normal(1., left_r, tile), [1.0, 0.]),
        (v3, get_tile_normal(bottom_r, 1., tile), [0.0, 1.]),
        (v4, get_tile_normal(1., 1., tile), [1.0, 1.]),
    ]);
    indices.append(&mut quad_cc_indices_off(idx_off));

    // // bottom right corner
    // let x = subdivs - 1.;
    // let y = subdivs - 1.;
    // // start in the top left of the bottom right
    // // first get the closest subdivision lines
    // // these could fall outside the square
    // let mut starting_x = (x * bottom_subdivs / subdivs).round();
    // let mut starting_y = (y * right_subdivs / subdivs).round();
    // // if they do fall outside, correct for it
    // if starting_x / bottom_subdivs <= x / subdivs {
    //     starting_x += 1.;
    // }
    // if starting_y / right_subdivs <= y / subdivs {
    //     starting_y += 1.;
    // }
    //
    // // in this loop we use xi1 as our "base" and xi0 as our "offset"
    // // we do the opposite in the previous loop so it's worth pointing out the shift in logic
    // for xi in (starting_x as u32)..(bottom_subdivs as u32 + 1) {
    //     for yi in (starting_y as u32)..(right_subdivs as u32 + 1) {
    //         let xi1 = xi as f32 / bottom_subdivs;
    //         let mut xi0 = (xi - 1) as f32 / bottom_subdivs;
    //         let yi1 = yi as f32 / right_subdivs;
    //         let mut yi0 = (yi - 1) as f32 / right_subdivs;
    //         if xi == starting_x as u32 {
    //             // if we're on the first iteration, then our previous is
    //             // actually OUR grid line, not the grid line of the adjacent
    //             xi0 = x / subdivs;
    //         }
    //         if yi == starting_y as u32 {
    //             yi0 = y / subdivs;
    //         }
    //         // we create the verts for now
    //         let mut v1 = [
    //             xi0 * TILE_WORLD_SIZE,
    //             get_tile_height(xi0, yi0, tile),
    //             yi0 * TILE_WORLD_SIZE,
    //         ];
    //         let mut v2 = [
    //             xi1 * TILE_WORLD_SIZE,
    //             get_tile_height(xi1, yi0, tile),
    //             yi0 * TILE_WORLD_SIZE,
    //         ];
    //         let mut v3 = [
    //             xi0 * TILE_WORLD_SIZE,
    //             get_tile_height(xi0, yi1, tile),
    //             yi1 * TILE_WORLD_SIZE,
    //         ];
    //         let mut v4 = [
    //             xi1 * TILE_WORLD_SIZE,
    //             get_tile_height(xi1, yi1, tile),
    //             yi1 * TILE_WORLD_SIZE,
    //         ];
    //         // now we correct the boundary verts
    //         if xi0 == x / bottom_subdivs {
    //             // if we're on the inner grid line, then the height of the vert
    //             // is going to be some combo of the height of the REAL corners
    //             // top left height
    //             let tlh = get_tile_height(x / subdivs, y / subdivs, tile);
    //             // bottom left height
    //             let blh = get_tile_height(x / subdivs, 1., tile);
    //             // we interpolate the height between the two corners
    //             v1[1] = lin_map32(y / subdivs, 1., tlh, blh, yi0);
    //             v3[1] = lin_map32(y / subdivs, 1., tlh, blh, yi1);
    //         }
    //         if yi0 == y / right_subdivs {
    //             // if we're on the inner grid line, then the height of the vert
    //             // is going to be some combo of the height of the REAL corners
    //             // top left height
    //             let tlh = get_tile_height(x / subdivs, y / subdivs, tile);
    //             // top right height
    //             let trh = get_tile_height(1., y / subdivs, tile);
    //             // we interpolate the height between the two corners
    //             v1[1] = lin_map32(x / subdivs, 1., tlh, trh, xi0);
    //             v2[1] = lin_map32(x / subdivs, 1., tlh, trh, xi1);
    //         }
    //         // now we can add the quads and verts as usual
    //         vertices.append(&mut vec![
    //             (v1, get_tile_normal(xi0, yi0, tile), [0.0, 0.]),
    //             (v2, get_tile_normal(xi1, yi0, tile), [1.0, 0.]),
    //             (v3, get_tile_normal(xi0, yi1, tile), [0.0, 1.]),
    //             (v4, get_tile_normal(xi1, yi1, tile), [1.0, 1.]),
    //         ]);
    //         indices.append(&mut cc_indices(cur_idx_set));
    //         cur_idx_set += 1;
    //     }
    // }
    //
    // // bottom left corner
    // let x = 0.;
    // let y = subdivs - 1.;
    // // start in the top left of the bottom right
    // // first get the closest subdivision lines
    // // these could fall outside the square
    // let mut ending_x = (x * bottom_subdivs / subdivs).round();
    // let mut starting_y = (y * left_subdivs / subdivs).round();
    // // if they do fall outside, correct for it
    // if ending_x / bottom_subdivs <= x / subdivs {
    //     ending_x += 1.;
    // }
    // if starting_y / left_subdivs <= y / subdivs {
    //     starting_y += 1.;
    // }
    //
    // // in this loop we use xi1 as our "base" and xi0 as our "offset"
    // // we do the opposite in the previous loop so it's worth pointing out the shift in logic
    // for xi in 1..(ending_x as u32 + 1) {
    //     for yi in (starting_y as u32)..(left_subdivs as u32 + 1) {
    //         let xi1 = xi as f32 / bottom_subdivs;
    //         let mut xi0 = (xi - 1) as f32 / bottom_subdivs;
    //         let yi1 = yi as f32 / left_subdivs;
    //         let mut yi0 = (yi - 1) as f32 / left_subdivs;
    //         if xi == ending_x as u32 {
    //             // if we're on the first iteration, then our previous is
    //             // actually OUR grid line, not the grid line of the adjacent
    //             xi0 = x / subdivs;
    //         }
    //         if yi == starting_y as u32 {
    //             yi0 = y / subdivs;
    //         }
    //         // we create the verts for now
    //         let mut v1 = [
    //             xi0 * TILE_WORLD_SIZE,
    //             get_tile_height(xi0, yi0, tile),
    //             yi0 * TILE_WORLD_SIZE,
    //         ];
    //         let mut v2 = [
    //             xi1 * TILE_WORLD_SIZE,
    //             get_tile_height(xi1, yi0, tile),
    //             yi0 * TILE_WORLD_SIZE,
    //         ];
    //         let mut v3 = [
    //             xi0 * TILE_WORLD_SIZE,
    //             get_tile_height(xi0, yi1, tile),
    //             yi1 * TILE_WORLD_SIZE,
    //         ];
    //         let mut v4 = [
    //             xi1 * TILE_WORLD_SIZE,
    //             get_tile_height(xi1, yi1, tile),
    //             yi1 * TILE_WORLD_SIZE,
    //         ];
    //         // now we correct the boundary verts
    //         if xi0 == x / bottom_subdivs {
    //             // if we're on the inner grid line, then the height of the vert
    //             // is going to be some combo of the height of the REAL corners
    //             // top right height
    //             let trh = get_tile_height(1. / subdivs, y / subdivs, tile);
    //             // bottom right height
    //             let brh = get_tile_height(1. / subdivs, 1., tile);
    //             // we interpolate the height between the two corners
    //             v2[1] = lin_map32(y / subdivs, 1., trh, brh, yi0);
    //             v4[1] = lin_map32(y / subdivs, 1., trh, brh, yi1);
    //         }
    //         if yi0 == y / left_subdivs {
    //             // if we're on the inner grid line, then the height of the vert
    //             // is going to be some combo of the height of the REAL corners
    //             // top right height
    //             let trh = get_tile_height(1. / subdivs, 1. / subdivs, tile);
    //             // top left height
    //             let tlh = get_tile_height(0., y / subdivs, tile);
    //             // we interpolate the height between the two corners
    //             v1[1] = lin_map32(x / subdivs, 1., tlh, trh, xi0);
    //             v2[1] = lin_map32(x / subdivs, 1., tlh, trh, xi1);
    //         }
    //         // now we can add the quads and verts as usual
    //         vertices.append(&mut vec![
    //             (v1, get_tile_normal(xi0, yi0, tile), [0.0, 0.]),
    //             (v2, get_tile_normal(xi1, yi0, tile), [1.0, 0.]),
    //             (v3, get_tile_normal(xi0, yi1, tile), [0.0, 1.]),
    //             (v4, get_tile_normal(xi1, yi1, tile), [1.0, 1.]),
    //         ]);
    //         indices.append(&mut cc_indices(cur_idx_set));
    //         cur_idx_set += 1;
    //     }
    // }

    // let mut p: Vec<[f32; 3]> = Vec::new();
    // let total = subdivs + top_subdivs;
    // let greater_divs = subdivs.max(top_subdivs);
    // let lesser_divs = subdivs.min(top_subdivs);
    // let cur_lesser = 1;
    // // helper var
    // let dims = tile.len();
    // // the heightmap y pixel is always 0 when doing the top
    // let yp = 0;
    // for x in 0..greater_divs+1 {
    //     if (cur_lesser as f32) / (lesser_divs as f32) < (x as f32) / (greater_divs as f32) {
    //         // if our current lesser-subdiv pos is less than the smaller subdivs,
    //         // then the next node must be a greater subdiv node
    //         // the node right after that will be a lesser subdiv node
    //         let xp =
    //             (lin_map(0., lesser_divs as f64, 0., dims as f64 - 1., cur_lesser as f64)).round() as usize;
    //         let z = f32::from_bits(tile[xp][yp]);
    //         p.push([cur_lesser as f32 / lesser_divs as f32, z, 0.])
    //         // now the next point must be a lesser subdiv node
    //     }
    //     let xp =
    //         (lin_map(0., greater_divs as f64, 0., dims as f64 - 1., x as f64)).round() as usize;
    //     let z = f32::from_bits(tile[xp][yp]);
    //     p.push([x as f32 / greater_divs as f32, z, 0.])
    // }
    //
    // // create the weird geometry around the edges
    // // top (-y) first
    // // first get all the points along this edge
    // let mut p: Vec<[f32; 3]> = Vec::new();
    // // the pixel that we take will always be one of the ones at the top
    // let yp = 0;
    // let dims = tile.len();
    // // do the points that we get from subdividing ourselves first
    // for x in 0..subdivs+1 {
    //     let xp =
    //         (lin_map(0., subdivs as f64, 0., dims as f64 - 1., x as f64)).round() as usize;
    //     let z = f32::from_bits(tile[xp][yp]);
    //     p.push([x as f32, z, 0.]);
    // }
    // // now the points that we get from subdividing the one above
    // for x in 0..top_subdivs+1 {
    //     let xp =
    //         (lin_map(0., top_subdivs as f64, 0., dims as f64 - 1., x as f64)).round() as usize;
    //     let z = f32::from_bits(tile[xp][yp]);
    //     p.push([x as f32, z, 0.]);
    // }
    // // now sort the points for ease of use
    // p.sort_by(|p1, p2| p1.partial_cmp(p2).unwrap());
    // // get the points across from these points that we would connect to otherwise
    //
    // // simple vertex positions
    // let mut v: Vec<[f32; 3]> = Vec::new();
    // let square_size = TILE_WORLD_SIZE as f32 / subdivs as f32;
    // for (x, y) in iproduct!(0..subdivs, 0..subdivs) {
    //     let dims = tile.len();
    //     let xp =
    //         (lin_map(0., subdivs as f64 - 1., 0., dims as f64 - 1., x as f64)).round() as usize;
    //     let yp =
    //         (lin_map(0., subdivs as f64 - 1., 0., dims as f64 - 1., y as f64)).round() as usize;
    //
    //     let z = f32::from_bits(tile[xp][yp]);
    //     v.push([
    //         x as f32 * square_size,
    //         z * square_size,
    //         y as f32 * square_size,
    //     ]);
    // }
    //
    // // next add extra verts that are created from differences in subdivisions between tiles
    // for x in 0..top_subdivs {
    //     v.push()
    // }

    let positions: Vec<_> = vertices.iter().map(|(p, _, _)| *p).collect();
    let normals: Vec<_> = vertices.iter().map(|(_, n, _)| *n).collect();
    let uvs: Vec<_> = vertices.iter().map(|(_, _, uv)| *uv).collect();

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    // mesh.compute_flat_normals();
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.set_indices(Some(Indices::U32(indices)));
    return mesh;
}
