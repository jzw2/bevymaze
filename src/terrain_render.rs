//use async_stream::__private::AsyncStream;
use std::collections::{HashMap, HashSet, VecDeque};
use std::f64::consts::PI;

//use async_stream::stream;
use bevy::ecs::reflect::ReflectComponent;
use bevy::math::{DVec2, DVec3, Vec2Swizzles, Vec4};
use bevy::prelude::{Component, Image, Mesh, Reflect, Resource};
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::render::render_resource::TextureFormat::{R32Float, Rg8Snorm, Rgba8Snorm};
use bevy::render::render_resource::{Extent3d, TextureDimension};
use delaunator::{prev_halfedge, triangulate, Triangulation};
use futures_util::{pin_mut, Stream, StreamExt};
use image::{ImageBuffer, Rgb, RgbImage};
use ordered_float::OrderedFloat;
use rand::{Rng, thread_rng};
// use rand::{thread_rng, Rng};
// use tokio::net::TcpStream;
// use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

use server::terrain_gen::{
    HeightMap, TerrainGenerator, TerrainNormals, TilePosition, MAX_HEIGHT, TILE_SIZE,
};
use server::util::{cart_to_polar, lin_map};

use crate::render::{CompleteVertices, SimpleVertex, VertexNormalUV};
//use crate::terrain_loader::get_height;

pub const TILE_WORLD_SIZE: f32 = TILE_SIZE as f32;
type TileOffset = TilePosition;

/// Number of chunks in each dir
pub const X_VIEW_DISTANCE: i64 = 30;
pub const Z_VIEW_DISTANCE: i64 = 250;
/// Number of meters in each dir
pub const X_VIEW_DIST_M: f64 = X_VIEW_DISTANCE as f64 * TILE_SIZE;
pub const Z_VIEW_DIST_M: f64 = Z_VIEW_DISTANCE as f64 * TILE_SIZE;
/// Number of vertices of our mesh
pub const TERRAIN_VERTICES: usize = 80000;

pub const PROB_TIGHTNESS: f64 = 100.0;

pub const SCALE: f64 = 0.15;
pub const TEXTURE_SCALE: f64 = 0.0001;

pub const LATTICE_GRID_SIZE: f64 = 1.0;

#[derive(Component, Default, Reflect)]
#[reflect(Component)]
pub struct MainTerrain;

/// Gets the height of the tile for a tile position
/// `xr` and `yr` are in the range `[0, 1]`
pub fn get_tile_height(xr: f32, yr: f32, height_map: &HeightMap) -> f32 {
    let dims = height_map.len() as f32 - 1.;
    return height_map[(xr * dims).round() as usize][(yr * dims).round() as usize] as f32;
}

pub fn get_tile_normal(xr: f32, yr: f32, normals: &TerrainNormals) -> [f32; 3] {
    let dims = normals.len() as f32 - 1.;
    return normals[(xr * dims).round() as usize][(yr * dims).round() as usize]
        .as_vec3()
        .to_array();
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

pub fn bary_poly_smooth_interp(
    ai: usize,
    bi: usize,
    ci: usize,
    height: &Vec<f32>,
    gradients: &Vec<f32>,
    bary: [f32; 2],
) -> f32 {
    let [x, y] = bary;

    let p10: f32 = height[ai];
    let p9 = gradients[ai * 2 + 1];
    let p8 = gradients[ai * 2];
    // let p7 = f32(0); // ignore, but here because it looks nicer
    let p2 = -2.0 * height[ci] + p9 + 2.0 * p10 + gradients[ci * 2 + 1];
    let p6 = height[ci] - p9 - p10 - p2;
    let p1 = -2.0 * height[bi] + p8 + 2.0 * p10 + gradients[bi * 2];
    let p5 = height[bi] - p8 - p10 - p1;
    let p4 = gradients[ci * 2] - p8;
    let p3 = gradients[bi * 2 + 1] - p9;

    return p1 * x*x*x + p2 * y*y*y + p3 * x*x * y + p4 * x * y*y +
            p5 * x*x + p6 * y*y + // p7 * x * y
            p8 * x + p9 * y +
            p10;
}

/// We create our lattice plane within an ellipse
/// After it's transformed this just to happens to be a decent approximation for a rectangle
/// Calculating the number of vertices in the ellipse is analogous to computing the area
/// which happens to have a nice formula (a * b * pi = A, where a/b are the major/minor axes)
/// So we calculate the length of the axes and then we use the formula x^2/a^2 + y^2/b^2 = 1
/// to get the specific height/width of the ellipse at a certain position along one of the axes
pub fn create_lattice_plane(
    approx_vertex_count: f64,
    x_view_distance: f64,
    z_view_distance: f64,
) -> Vec<DVec2> {
    let mut verts: HashSet<[OrderedFloat<f64>; 2]> = HashSet::new();
    let x_bound = (x_view_distance * SCALE).asinh() / SCALE;
    let z_bound = (z_view_distance * SCALE).asinh() / SCALE;
    let aspect_ratio = x_bound / z_bound;

    let z_0_sqr: f64 = 3. / 4.;
    let z_0 = z_0_sqr.sqrt();

    let x_verts = (z_0 * approx_vertex_count * aspect_ratio / PI)
        .sqrt()
        .ceil() as i32;
    let z_verts = (approx_vertex_count / (PI * x_verts as f64)).ceil() as i32;

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
            z_idx as f64 + thread_rng().gen_range(0.25..0.75),
        );
        let x_bound_cur = (x_bound.powi(2) - aspect_ratio.powi(2) * z_pos.powi(2)).sqrt();
        let x_verts_above = (x_bound_cur / x_bound * x_verts as f64).ceil() as i32;

        for x_idx in -x_verts_above..x_verts_above {
            let x_pos = lin_map(
                -x_verts as f64,
                x_verts as f64,
                -x_bound,
                x_bound,
                x_idx as f64 + thread_rng().gen_range(0.25..0.75)/* * (z_idx % 2) as f64*/,
            );
            verts.insert([OrderedFloat(x_pos), OrderedFloat(z_pos)]);
        }
    }

    return verts
        .into_iter()
        .map(|v| DVec2::new(v[0].0, v[1].0))
        .collect();
}

/// Here we sample assuming the probability dist
/// p = 1/sqrt(r^2+1)
/// the cdf of p aka c is c = asinh(r)
/// the inverse cdf is then sinh(p)
/// This makes our math look quite simple
/// The most complicated bit is changing to polar coordinates to do the transformation
pub fn transform_lattice_positions(lattice: &mut Vec<DVec2>, rounding: Option<f64>) {
    let mut verts: HashSet<[OrderedFloat<f64>; 2]> = HashSet::new();
    for lattice_pos in lattice.iter_mut() {
        let mag = lattice_pos.length();
        if mag != 0. {
            let scale_fac = ((mag * SCALE).sinh() / SCALE) / mag;
            *lattice_pos *= scale_fac;
        }
        // round the transformed pos to the nearest grid pos (quarter of a meter)
        if let Some(rounding) = rounding {
            *lattice_pos = (*lattice_pos / rounding)/*.round()*/ * rounding;
        }
        verts.insert([OrderedFloat(lattice_pos.x), OrderedFloat(lattice_pos.y)]);
    }

    lattice.clear();
    for v in verts {
        lattice.push(DVec2::new(v[0].0, v[1].0));
    }
}

pub fn length_order_verts(verts: &mut Vec<DVec2>) {
    verts.sort_by(|v1, v2| {
        v1.length_squared()
            .partial_cmp(&v2.length_squared())
            .unwrap()
    });
}

/// Order the points into a hilbert curve
/// Unfortunately this is lossy due to the conversion between fixed point and floating point
/// There's a way to make it not lossy by storing the verts in a hash map, sorting the keys, and then unhashing them later, but it shouldn't matter much because
/// the loss is sub-millimeter
pub fn hilbert_order_verts(verts: &mut Vec<DVec2>, x_view_dist: f64, z_view_dist: f64) {
    let mut hilbert_verts: Vec<hilbert::Point> =
        vec![hilbert::Point::new(0, &mut [0, 0]); verts.len()];
    const BOUND: f64 = (1 << (31 - 1)) as f64;
    // TODO: figure out a way to calculate this for real, I just guestimated this from looking at a graph
    const EXTRA: f64 = 2.8;
    for idx in 0..verts.len() {
        let vert = verts[idx];
        let x_idx = lin_map(
            -x_view_dist * EXTRA,
            x_view_dist * EXTRA,
            0.0,
            BOUND,
            vert.x,
        )
        .clamp(0., BOUND)
        .round() as u32;
        let y_idx = lin_map(
            -z_view_dist * EXTRA,
            z_view_dist * EXTRA,
            0.0,
            BOUND,
            vert.y,
        )
        .clamp(0., BOUND)
        .round() as u32;
        hilbert_verts[idx] = hilbert::Point::new(idx, &[x_idx, y_idx]);
    }
    hilbert::Point::hilbert_sort(&mut hilbert_verts, 32);
    for idx in 0..hilbert_verts.len() {
        let h_vert: &hilbert::Point = &hilbert_verts[idx];
        let x: f64 = lin_map(
            0.0,
            BOUND,
            -x_view_dist * EXTRA,
            x_view_dist * EXTRA,
            h_vert.get_coordinates()[0] as f64,
        );
        let z: f64 = lin_map(
            0.0,
            BOUND,
            -z_view_dist * EXTRA,
            z_view_dist * EXTRA,
            h_vert.get_coordinates()[1] as f64,
        );
        verts[idx] = DVec2::new(x, z);
    }
}

pub fn angle_sort_verts(verts: &mut Vec<DVec2>) {
    verts.sort_by(|v1, v2| {
        let mut v1_val = v1.y.atan2(v1.x);
        let mut v2_val = v2.y.atan2(v2.x);
        if v1_val.is_nan() {
            v1_val = 0.
        }
        if v2_val.is_nan() {
            v2_val = 0.
        }
        v1_val.partial_cmp(&v2_val).unwrap()
    });
}

pub fn create_base_lattice() -> Vec<DVec2> {
    return create_base_lattice_with_verts(TERRAIN_VERTICES as f64);
}

pub fn create_base_lattice_with_verts(approx_vertex_count: f64) -> Vec<DVec2> {
    return create_lattice(approx_vertex_count, X_VIEW_DIST_M, Z_VIEW_DIST_M);
}

pub fn create_lattice(approx_vertex_count: f64, x_view_dist: f64, z_view_dist: f64) -> Vec<DVec2> {
    let mut verts = create_lattice_plane(approx_vertex_count, x_view_dist, z_view_dist);
    transform_lattice_positions(&mut verts, Some(LATTICE_GRID_SIZE));
    //hilbert_order_verts(&mut verts, x_view_dist, z_view_dist);
    angle_sort_verts(&mut verts);
    length_order_verts(&mut verts);
    return verts;
}

fn fix_quads(triangulation: &mut Triangulation, verts: &Vec<DVec2>) {
    // we iterate over all the triangles and fix the quads
    // let mut fixed = HashSet::<usize>::new();
    for tri in 0..triangulation.len() {
        for (ao, bo, co) in [(0, 1, 2), (2, 0, 1), (1, 2, 0)] {
            let (a, b, c) = (3 * tri + ao, 3 * tri + bo, 3 * tri + co);
            let (ai, bi, ci) = (
                triangulation.triangles[a],
                triangulation.triangles[b],
                triangulation.triangles[c],
            );
            let (av, bv, cv) = (verts[ai], verts[bi], verts[ci]);
            if (av - bv).dot(DVec2::Y) == 0.
                && (bv - cv).dot(DVec2::X) == 0.
                && (av - bv).length_squared() == (bv - cv).length_squared()
                && av.y < bv.y
                && bv.x < cv.x
            {
                // first check if we have a bottom-left right triangle
                // now check  if we have top-right right triangle to match
                if triangulation.halfedges[c] < triangulation.triangles.len() {
                    // case 1: a top left, b bottom left, c bottom right
                    let d_tri = triangulation.halfedges[c] / 3;
                    let di = triangulation.triangles[prev_halfedge(triangulation.halfedges[c])];
                    let dv = verts[di];
                    if (av - dv).dot(DVec2::X) == 0.
                        && (cv - dv).dot(DVec2::Y) == 0.
                        && av.x < dv.x
                        && dv.y < cv.y
                    {
                        triangulation.triangles[a] = ai;
                        triangulation.triangles[b] = bi;
                        triangulation.triangles[c] = di;

                        triangulation.triangles[3 * d_tri] = bi;
                        triangulation.triangles[3 * d_tri + 1] = ci;
                        triangulation.triangles[3 * d_tri + 2] = di;
                        break;
                    }
                }
            }
        }
    }
}

pub fn create_triangulation(lattice: Option<Vec<DVec2>>) -> (Triangulation, Vec<DVec2>) {
    let verts: Vec<DVec2>;
    if let Some(provided_verts) = lattice {
        verts = provided_verts;
    } else {
        verts = create_base_lattice();
    }

    // The actual vertices that we will put into the mesh
    let delauney_verts: Vec<delaunator::Point> = verts
        .iter()
        .map(|e| delaunator::Point::from(e.to_array()))
        .collect();

    let mut triangulation = triangulate(&delauney_verts);

    //fix_quads(&mut triangulation, &verts);

    return (triangulation, verts);
}

/// Create a terrain mesh but without baked heights
pub fn create_terrain_mesh(triangulation: &Triangulation, verts: &Vec<DVec2>, max_dist: f64) -> Mesh {
    // iterate over the triangles, and remove any OOB triangles
    let mut triangles: Vec<[usize; 3]> = vec![];
    for tri in 0..triangulation.len() {
        triangles.push([triangulation.triangles[3*tri], triangulation.triangles[3*tri + 1], triangulation.triangles[3*tri + 2]]);
    }
    let max_dist_sqr = max_dist * max_dist;
    let indices: Vec<u32> = triangles.iter().filter_map(|t | {
        for v in t {
            if verts[*v].length_squared() < max_dist_sqr {
                return Some([t[0] as u32, t[1] as u32, t[2] as u32]);
            }
        }
        return None;
    }).flatten().collect();

    let positions: Vec<_> = verts
        .iter()
        .map(|e| {
            let mut v = e.xxy().as_vec3();
            v.y = 0.;
            v
        })
        .collect();

    let mut terrain_mesh = Mesh::new(PrimitiveTopology::TriangleList);
    terrain_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    terrain_mesh.set_indices(Some(Indices::U32(indices)));
    return terrain_mesh;
}

pub fn create_terrain_normal_map(generator: &TerrainGenerator) -> Image {
    let x_bound = /*X_VIEW_DIST_M;*/ (X_VIEW_DIST_M * TEXTURE_SCALE).asinh() / TEXTURE_SCALE;
    let z_bound = /*Z_VIEW_DIST_M;*/ (Z_VIEW_DIST_M * TEXTURE_SCALE).asinh() / TEXTURE_SCALE;

    let dimx = 1024;
    let dimz = (dimx as f64 * (z_bound / x_bound)).round() as usize;

    println!("{} {} | {} {}", x_bound, z_bound, dimx, dimz);

    let mut data = vec![vec![DVec3::splat(0.); dimz]; dimx];
    // choose 4 points per pixel to sample from, hence * 2
    // s means "sample" per pixel dimension, so s*s samples total per pixel
    let s = 1;
    for x in 0..dimx * s {
        for z in 0..dimz * s {
            // get the world pos
            // This is a map from normalized texture space -> unnormalized texture space -> world space
            // Normalized texture space is bounded by `dim`, unnormalized texture space is bounded by `dim_bound`
            // world space is bounded by `Z_VIEW_DISTANCE.max(X_VIEW_DISTANCE)`
            let mut x_world_pos = lin_map(0., (dimx * s) as f64, -x_bound, x_bound, x as f64);
            let mut z_world_pos = lin_map(0., (dimz * s) as f64, -z_bound, z_bound, z as f64);
            let r = (x_world_pos.powi(2) + z_world_pos.powi(2)).sqrt();
            let theta = z_world_pos.atan2(x_world_pos);
            x_world_pos = (r * TEXTURE_SCALE).sinh() / TEXTURE_SCALE * theta.cos();
            z_world_pos = (r * TEXTURE_SCALE).sinh() / TEXTURE_SCALE * theta.sin();
            // now get the normal vector for this one
            let normal = generator.get_normal(x_world_pos, z_world_pos);
            // finally average it with the existing stored vec
            let x_idx = x / s;
            let y_idx = z / s;
            data[x_idx][y_idx] += normal / (s * s) as f64;
        }
        println!("Done row {x}");
    }

    // we store the vectors with 2 bytes
    // we omit the last dimension because we can recalculate it in the shader
    // since we know that magnitude of the vec is 1, meaning x^2+y^2+z^2=1 and the
    // sign of y is always 1.
    // we flatten the data and convert to our compressed format
    let dataclone = data.clone();
    let mut simple_data: Vec<u8> = vec![];
    for z in 0..dimz {
        for x in 0..dimx {
            let d = data[x][z];
            let x = lin_map(-1., 1., -128.0, 127.0, d.x).round() as i8;
            simple_data.push(x as u8);
            let z = lin_map(-1., 1., -128.0, 127.0, d.z).round() as i8;
            simple_data.push(z as u8);
            // simple_data.push(0);
            // simple_data.push(0);
        }
    }

    // DEBUG
    let mut img: RgbImage = ImageBuffer::new(dimx as u32, dimz as u32);
    for x in 0..img.width() {
        for y in 0..img.height() {
            let d = dataclone[x as usize][y as usize];
            let xv = lin_map(-1., 1., 0., 255., d.x).round() as u8;
            let zv = lin_map(-1., 1., 0., 255., d.z).round() as u8;
            let yv = lin_map(0., 1., 0., 255., d.y).round() as u8;
            img.put_pixel(x, y, Rgb([xv, yv, zv]));
        }
        println!("Debug done row {x}")
    }
    img.save("normal_out.png").unwrap();

    return Image::new(
        Extent3d {
            width: dimx as u32,
            height: dimz as u32,
            depth_or_array_layers: 1,
        },
        TextureDimension::D2,
        simple_data,
        Rg8Snorm,
    );
}
