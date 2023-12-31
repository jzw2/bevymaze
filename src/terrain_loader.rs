use crate::terrain_render::{
    create_lattice_plane, transform_lattice_positions, TERRAIN_VERTICES, X_VIEW_DISTANCE,
    Z_VIEW_DISTANCE,
};
use bevy::prelude::Mesh;
use bevy_easings::Lerp;
use futures_util::{FutureExt, SinkExt, Stream, StreamExt, TryFutureExt};
use log::*;
use server::terrain_data::TerrainTile;
use server::terrain_gen::TILE_SIZE;
use server::util::{cart_to_polar, lin_map, polar_to_cart};
use std::arch::asm;
use std::f64::consts::PI;
use std::io::BufReader;
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{
    connect_async,
    tungstenite::{Error, Result},
    MaybeTlsStream, WebSocketStream,
};
use url::Url;

use kiddo::KdTree;
use kiddo::NearestNeighbour;
use kiddo::SquaredEuclidean;

struct TileData {
    data: Vec<Vec<f64>>,
}

/// We get the height of the terrain at a certain point based off of the LOD
/// We use bilinear filtering
/// We calculate the appropriate LOD based off the density for a particular section
pub async fn get_height(
    x: f64,
    z: f64,
    center_coords: (i32, i32),
    socket: &mut WebSocketStream<MaybeTlsStream<TcpStream>>,
) -> f64 {
    // The coordinates of the point's chunk
    let chunk_x = (x / TILE_SIZE).floor();
    let chunk_z = (z / TILE_SIZE).floor();
    let chunk = get_chunk((chunk_x as i32, chunk_z as i32), center_coords, socket).await;
    return chunk.get_height(x - chunk_x * TILE_SIZE, z - chunk_z * TILE_SIZE);
}

/// We get the height of the terrain at a certain point based off of the LOD
/// We use bilinear filtering
/// We calculate the appropriate LOD based off the density for a particular section
pub fn get_chunk_height(x: f64, z: f64, chunk: TerrainTile) -> f64 {
    // The coordinates of the point's chunk
    let chunk_x = (x / TILE_SIZE).floor();
    let chunk_z = (z / TILE_SIZE).floor();
    return chunk.get_height(x - chunk_x * TILE_SIZE, z - chunk_z * TILE_SIZE);
}

/// Gets the area of this chunk in lattice space
/// Used to get the LOD level because the area is proportional to the number of lattice points
fn get_lattice_space_area(chunk_coords: (i32, i32), center_coords: (i32, i32)) -> f64 {
    let normalized_coordinates = (
        chunk_coords.0 - center_coords.0,
        chunk_coords.1 - center_coords.1,
    );
    return match normalized_coordinates {
        // the "origin" chunks are best approximated by quarters of circles with a little extra
        (0, 0) | (0, -1) | (-1, -1) | (-1, 0) => {
            PI / 8. * (7. * TILE_SIZE.asinh().powi(2) - (TILE_SIZE * 2.0f64.sqrt()).asinh().powi(2))
        }
        n_c => {
            // first get the vertices of the chunk/tile
            let p1 = (TILE_SIZE * n_c.0 as f64, TILE_SIZE * n_c.1 as f64);
            let p2 = (TILE_SIZE * (n_c.0 + 1) as f64, TILE_SIZE * n_c.1 as f64);
            let p3 = (
                TILE_SIZE * (n_c.0 + 1) as f64,
                TILE_SIZE * (n_c.1 + 1) as f64,
            );
            let p4 = (TILE_SIZE * n_c.0 as f64, TILE_SIZE * (n_c.1 + 1) as f64);
            // now transform to polar form
            let p1 = cart_to_polar(p1);
            let p2 = cart_to_polar(p2);
            let p3 = cart_to_polar(p3);
            let p4 = cart_to_polar(p4);
            // map from tile space to lattice space & back to cartesian rep
            let p1 = polar_to_cart((p1.0.asinh(), p1.1));
            let p2 = polar_to_cart((p2.0.asinh(), p2.1));
            let p3 = polar_to_cart((p3.0.asinh(), p3.1));
            let p4 = polar_to_cart((p4.0.asinh(), p4.1));
            // finally we use the shoelace formula to get the area
            fn det(p1: &(f64, f64), p2: &(f64, f64)) -> f64 {
                // treat p1 and p2 like column vectors
                p1.0 * p2.1 - p1.1 * p2.0
            }
            0.5 * (det(&p1, &p2) + det(&p2, &p3) + det(&p3, &p4) + det(&p4, &p1))
        }
    };
}

fn get_lod(chunk_coords: (i32, i32), center_coords: (i32, i32)) -> u32 {
    // we first get the lattice points per unit of area
    let x_bound = (X_VIEW_DISTANCE as f64 * TILE_SIZE).asinh();
    let z_bound = (Z_VIEW_DISTANCE as f64 * TILE_SIZE).asinh();
    let total_area = x_bound * z_bound * PI; // area of an ellipse a*b*pi
    let verts_per_area = TERRAIN_VERTICES as f64 / total_area;
    // now get our number of vertices
    let verts = get_lattice_space_area(chunk_coords, center_coords) * verts_per_area;
    // finally select the LOD
    // make sure the LOD has around twice the number of data points as the number of verts we chose
    return (10. - 0.5 * (2. * verts).log2()).min(10.).max(0.).ceil() as u32;
}

pub async fn get_chunk(
    chunk_coords: (i32, i32),
    center_coords: (i32, i32),
    socket: &mut WebSocketStream<MaybeTlsStream<TcpStream>>,
) -> TerrainTile {
    let lod = get_lod(chunk_coords, center_coords);
    socket
        .send(Message::Text(format!(
            "{},{}:{lod}",
            chunk_coords.0, chunk_coords.1
        )))
        .await
        .expect("Failed to send for chunk");
    let msg = socket.next().await.expect("Can't fetch chunk");
    return TerrainTile::from(msg.unwrap().into_data());
}

pub fn get_required_data(
    center: (i32, i32),
    lattice: &Vec<(i32, i32)>,
    data: &mut KdTree<f64, 2>,
) -> Vec<(i32, i32)> {
    let mut to_fetch = vec![];
    /*
    // iterate over all the vertices in our terrain mesh
    // find the closest point
    // if it's not close enough, fetch a new one
    for lattice_pos in lattice {
        let world_pos = (center.0 + lattice_pos.0, center.1 + lattice_pos.1);
        let nearest = data.nearest_one::<SquaredEuclidean>(&[world_pos.0, world_pos.1]);
        // TODO: make non-static distance
        if nearest.distance > 1.0 {
            to_fetch.push(world_pos);
        }
    }*/
    return to_fetch;
}
