use crate::render::{
    quad_cc_indices, quad_cc_indices_off, quad_cw_indices_off, tri_cc_indices_off,
    tri_cw_indices_off, CompleteVertices,
};
use bevy::math::Vec3;
use bevy::prelude::Mesh;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use itertools::iproduct;
use lazy_static::lazy_static;
use server::terrain_gen::{HeightMap, TerrainNormals, Tile, MAX_HEIGHT, TILE_SIZE};
use server::util::{lin_map, lin_map32};
use std::mem::swap;
use std::os::unix::raw::time_t;

pub const TILE_WORLD_SIZE: f32 = TILE_SIZE as f32;
const BASE_VERTICES: f64 = 8182.*4.;
const FREE_SUBDIVISIONS: f64 = 2.;
lazy_static! {
    pub static ref BASE_SUBDIVISIONS: f64 = BASE_VERTICES.sqrt();
}

type TileOffset = (i64, i64);

fn get_tile_dist(t1_off: TileOffset, t2_off: TileOffset) -> f64 {
    let d = (t1_off.0 - t2_off.0, t1_off.1 - t2_off.1);
    return ((d.0 * d.0 + d.1 * d.1) as f64).sqrt();
}

fn get_side_subdivs(d: f64) -> f32 {
    return ((*BASE_SUBDIVISIONS / ((d + 1.)*(d + 1.))) + FREE_SUBDIVISIONS).round() as f32;
}

/// Gets the height of the tile for a tile position
/// `xr` and `yr` are in the range `[0, 1]`
pub fn get_tile_height(xr: f32, yr: f32, height_map: &HeightMap) -> f32 {
    let dims = height_map.len() as f32 - 1.;
    return height_map[(xr * dims).round() as usize][(yr * dims).round() as usize];
}

pub fn get_tile_normal(xr: f32, yr: f32, normals: &TerrainNormals) -> [f32; 3] {
    let dims = normals.len() as f32 - 1.;
    return normals[(xr * dims).round() as usize][(yr * dims).round() as usize].to_array();
}

/// Get the color of a vertex
/// this is essentially the material
pub fn vertex_color(vertex: [f32; 3]) -> [f32; 4] {
    // first get the height as a fraction of total possible height
    let height_frac = (vertex[1] as f64 / MAX_HEIGHT) as f32;
    if height_frac < 0.3 {
        return [0.1, 0.5, 0.2, 1.];
    } else if height_frac < 0.5 {
        return [0.2, 0.6, 0.25, 1.];
    } else if height_frac > 0.8 {
        return [0.95, 0.95, 0.95, 1.];
    }

    return [0.2, 0.2, 0.2, 1.];
}

/// Gets the mesh for a tile
/// Fixes cracks between tiles by being smart about geometry
pub fn get_tile_mesh(tile_offset: TileOffset, tile: &Tile) -> Mesh {
    let height_map = &tile.0;
    let normals = &tile.1;
    let subdivs = get_side_subdivs(get_tile_dist(tile_offset, (0, 0)));
    let bottom_subdivs = get_side_subdivs(get_tile_dist(tile_offset, (0, -1)));
    let right_subdivs = get_side_subdivs(get_tile_dist(tile_offset, (-1, 0)));

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
                get_tile_height(xi0, yi0, height_map),
                yi0 * TILE_WORLD_SIZE,
            ];
            let v2 = [
                xi1 * TILE_WORLD_SIZE,
                get_tile_height(xi1, yi0, height_map),
                yi0 * TILE_WORLD_SIZE,
            ];
            let v3 = [
                xi0 * TILE_WORLD_SIZE,
                get_tile_height(xi0, yi1, height_map),
                yi1 * TILE_WORLD_SIZE,
            ];
            let v4 = [
                xi1 * TILE_WORLD_SIZE,
                get_tile_height(xi1, yi1, height_map),
                yi1 * TILE_WORLD_SIZE,
            ];
            vertices.append(&mut vec![
                (v1, get_tile_normal(xi0, yi0, normals), [0.0, 0.]),
                (v2, get_tile_normal(xi1, yi0, normals), [1.0, 0.]),
                (v3, get_tile_normal(xi0, yi1, normals), [0.0, 1.]),
                (v4, get_tile_normal(xi1, yi1, normals), [1.0, 1.]),
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
        let mut x0r = xi as f32 / greater_subdivs;
        let x1r = (xi + 1) as f32 / greater_subdivs;
        let mut c0r = (xi as f32 * lesser_subdivs / greater_subdivs).round() / lesser_subdivs;
        let c1r = ((xi + 1) as f32 * lesser_subdivs / greater_subdivs).round() / lesser_subdivs;
        let mut v1 = [[
            x0r * TILE_WORLD_SIZE,
            get_tile_height(x0r, y0r, height_map),
            y0r * TILE_WORLD_SIZE,
        ], get_tile_normal(x0r, y0r, normals)];
        let mut v2 = [[
            x1r * TILE_WORLD_SIZE,
            get_tile_height(x1r, y0r, height_map),
            y0r * TILE_WORLD_SIZE,
        ], get_tile_normal(x1r, y0r, normals)];
        let mut v3 = [[
            c0r * TILE_WORLD_SIZE,
            get_tile_height(c0r, y1r, height_map),
            y1r * TILE_WORLD_SIZE,
        ], get_tile_normal(c0r, y1r, normals)];
        if c0r != c1r {
            let mut v4 = [[
                c1r * TILE_WORLD_SIZE,
                get_tile_height(c1r, y1r, height_map),
                y1r * TILE_WORLD_SIZE,
            ], get_tile_normal(c1r, y1r, normals)];
            // we swap the here to keep the quads consistent
            // only swap if we swapped the y positions previous
            if subdivs <= bottom_subdivs {
                swap(&mut v1, &mut v3);
                swap(&mut v2, &mut v4);
            }
            vertices.append(&mut vec![
                (v1[0], v1[1], [0.0, 0.]),
                (v2[0], v2[1], [1.0, 0.]),
                (v3[0], v3[1], [0.0, 1.]),
                (v4[0], v4[1], [1.0, 1.]),
            ]);
            indices.append(&mut quad_cc_indices_off(idx_off));
            idx_off += 4;
        } else {
            vertices.append(&mut vec![
                (v1[0], v1[1], [0.0, 0.]),
                (v2[0], v2[1], [1.0, 0.]),
                (v3[0], v3[1], [0.0, 1.]),
            ]);
            if subdivs <= bottom_subdivs {
                indices.append(&mut quad_cw_indices_off(idx_off));
            } else {
                indices.append(&mut quad_cc_indices_off(idx_off));
            }
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
        let mut v1 = [[
            x0r * TILE_WORLD_SIZE,
            get_tile_height(x0r, y0r, height_map),
            y0r * TILE_WORLD_SIZE,
        ], get_tile_normal(x0r, y0r, normals)];
        let mut v2 = [[
            x1r * TILE_WORLD_SIZE,
            get_tile_height(x1r, c0r, height_map),
            c0r * TILE_WORLD_SIZE,
        ], get_tile_normal(x1r, c0r, normals)];
        let mut v3 = [[
            x0r * TILE_WORLD_SIZE,
            get_tile_height(x0r, y1r, height_map),
            y1r * TILE_WORLD_SIZE,
        ], get_tile_normal(x0r, y1r, normals)];
        if c0r != c1r {
            let mut v4 = [[
                x1r * TILE_WORLD_SIZE,
                get_tile_height(x1r, c1r, height_map),
                c1r * TILE_WORLD_SIZE,
            ], get_tile_normal(x1r, c1r, normals)];
            // we swap the here to keep the quads consistent
            // only swap if we swapped the x positions previous
            if subdivs <= right_subdivs {
                swap(&mut v1, &mut v2);
                swap(&mut v3, &mut v4);
            }
            vertices.append(&mut vec![
                (v1[0], v1[1], [0.0, 0.]),
                (v2[0], v2[1], [1.0, 0.]),
                (v3[0], v3[1], [0.0, 1.]),
                (v4[0], v4[1], [1.0, 1.]),
            ]);
            indices.append(&mut quad_cc_indices_off(idx_off));
            idx_off += 4;
        } else {
            vertices.append(&mut vec![
                (v1[0], v1[1], [0.0, 0.]),
                (v2[0], v2[1], [1.0, 0.]),
                (v3[0], v3[1], [0.0, 1.]),
            ]);
            if subdivs <= right_subdivs {
                indices.append(&mut quad_cw_indices_off(idx_off));
            } else {
                indices.append(&mut quad_cc_indices_off(idx_off));
            }
            idx_off += 3;
        }
    }

    // lastly we do the bottom left corner
    let v1r = (subdivs - 1.) / subdivs;
    let v1 = [
        v1r * TILE_WORLD_SIZE,
        get_tile_height(v1r, v1r, height_map),
        v1r * TILE_WORLD_SIZE,
    ];
    let left_r = (right_subdivs - 1.) / right_subdivs;
    let v2 = [
        TILE_WORLD_SIZE,
        get_tile_height(1., left_r, height_map),
        left_r * TILE_WORLD_SIZE,
    ];
    let bottom_r = (bottom_subdivs - 1.) / bottom_subdivs;
    let v3 = [
        bottom_r * TILE_WORLD_SIZE,
        get_tile_height(bottom_r, 1., height_map),
        TILE_WORLD_SIZE,
    ];
    let v4 = [
        TILE_WORLD_SIZE,
        get_tile_height(1., 1., height_map),
        TILE_WORLD_SIZE,
    ];
    vertices.append(&mut vec![
        (v1, get_tile_normal(v1r, v1r, normals), [0.0, 0.]),
        (v2, get_tile_normal(1., left_r, normals), [1.0, 0.]),
        (v3, get_tile_normal(bottom_r, 1., normals), [0.0, 1.]),
        (v4, get_tile_normal(1., 1., normals), [1.0, 1.]),
    ]);
    indices.append(&mut quad_cc_indices_off(idx_off));

    let positions: Vec<_> = vertices.iter().map(|(p, _, _)| *p).collect();
    let normals: Vec<_> = vertices.iter().map(|(_, n, _)| *n).collect();
    let uvs: Vec<_> = vertices.iter().map(|(_, _, uv)| *uv).collect();

    // now we can grab the COLORS
    // color is primarily based on height

    let colors: Vec<[f32; 4]> = positions
        .iter()
        .map(|[x, y, z]| vertex_color([*x, *y, *z]))
        .collect();

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    // mesh.compute_flat_normals();
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.set_indices(Some(Indices::U32(indices)));
    // mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    return mesh;
}
