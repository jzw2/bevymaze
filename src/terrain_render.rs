use crate::render::{
    quad_cc_indices, quad_cc_indices_off, quad_cw_indices_off, tri_cc_indices, tri_cc_indices_off,
    tri_cw_indices_off, CompleteVertices, SimpleVertex,
};
use bevy::math::{DVec3, Vec3, Vec4};
use bevy::prelude::system_adapter::new;
use bevy::prelude::Mesh;
use bevy::render::mesh::VertexAttributeValues::Float32x2;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use delaunator::{triangulate, Point};
use image::imageops::FilterType;
use image::DynamicImage;
use itertools::iproduct;
use lazy_static::lazy_static;
use rand::{thread_rng, Rng};
use server::terrain_gen::{
    HeightMap, TerrainGenerator, TerrainNormals, TileLODDescription, TilePosition, TileSignal,
    MAX_HEIGHT, TILE_RESOLUTION, TILE_SIZE,
};
use server::util::{lin_map, lin_map32};
use std::f64::consts::PI;
use std::mem::swap;
use std::os::unix::raw::time_t;

pub const TILE_WORLD_SIZE: f32 = TILE_SIZE as f32;
const BASE_VERTICES: f64 = 8182.;
const FREE_SUBDIVISIONS: f64 = 2.;
lazy_static! {
    pub static ref BASE_SUBDIVISIONS: f64 = BASE_VERTICES.sqrt();
}
const MIN_RESOLUTION: f64 = 16.;

type TileOffset = TilePosition;

/// Number of chunks in each dir
pub const X_VIEW_DISTANCE: i64 = 20;
pub const Z_VIEW_DISTANCE: i64 = 200;
/// Number of vertices of our mesh
pub const MAX_VERTICIES: i64 = 100000;

pub const VIEW_DISTANCE: i64 = 150;

pub const PROB_TIGHTNESS: f64 = 100.0;

pub fn get_tile_dist(t1_off: TileOffset, t2_off: TileOffset) -> f64 {
    let d = (t1_off.0 - t2_off.0, t1_off.1 - t2_off.1);
    return ((d.0 * d.0 + d.1 * d.1) as f64).sqrt();
}

fn get_side_subdivs(d: f64) -> f32 {
    // return 2.;
    return ((*BASE_SUBDIVISIONS / (d + 1.)) + FREE_SUBDIVISIONS).round() as f32;
}

fn get_tile_resolution(d: f64) -> f64 {
    return TILE_RESOLUTION as f64;
    let d = 0.01 * d;
    return ((TILE_RESOLUTION as f64 - MIN_RESOLUTION) / (d + 1.) + MIN_RESOLUTION).round();
}

/// Get a new image scaled for the distance specified
pub fn get_scaled_normal_map(normals: DynamicImage, tile_dist: f64) -> DynamicImage {
    let new_res = get_tile_resolution(tile_dist) as u32;
    let new_img = normals.resize_exact(new_res, new_res, FilterType::Nearest);
    println!("dist {} new {}", tile_dist, new_res);
    new_img.save(format!("{}.png", tile_dist));
    return new_img;
}

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

pub fn create_lattice_plane() -> (Vec<DVec3>, Vec<u32>, Vec<Vec4>) {
    let mut indices: Vec<u32> = vec![];
    let mut verts: Vec<DVec3> = vec![];
    let mut colors: Vec<Vec4> = vec![];
    let mut delauney_verts: Vec<Point> = vec![];
    let x_bound = (X_VIEW_DISTANCE as f64 * TILE_SIZE).asinh();
    let z_bound = (Z_VIEW_DISTANCE as f64 * TILE_SIZE).asinh();
    let aspect_ratio = x_bound / z_bound;

    let z_0_sqr: f64 = 3. / 4.;
    let z_0 = z_0_sqr.sqrt();

    let x_verts = (z_0 * MAX_VERTICIES as f64 * aspect_ratio / PI)
        .sqrt()
        .ceil() as i32;
    let z_verts = (MAX_VERTICIES as f64 / (PI * x_verts as f64)).ceil() as i32;

    // let aspect_ratio = (x_bound / z_bound) as f64;
    // let z_0 = 3.0f64.sqrt() / 2.0;
    // let x_verts = (MAX_VERTICIES as f64 / aspect_ratio * z_0).sqrt().ceil() as i32;
    // let z_verts = (aspect_ratio / z_0 * x_verts as f64).ceil() as i32;
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

        let green = thread_rng().gen_range(0.0..1.0);
        let blue = thread_rng().gen_range(0.0..1.0);
        for x_idx in -x_verts_above..x_verts_above {
            let x_pos = lin_map(
                -x_verts as f64,
                x_verts as f64,
                -x_bound,
                x_bound,
                x_idx as f64 + 0.5 * (z_idx % 2) as f64,
            );
            verts.push(DVec3::new(x_pos + 0.01, 0., z_pos + 0.01));
            colors.push(Vec4::new(1., green, 1. - blue, 1.));
            delauney_verts.push(Point { x: x_pos, y: z_pos });
            // now add the verts
            // if x_idx < x_verts - 1 {
            //     // trying to add stuff at the last vert gets wonky fast,
            //     // so DON'T DO IT
            //     let i = x_idx + z_idx * x_verts;
            //     if z_idx % 2 == 0 {
            //         // we add the triangles at i, i+1, i - x and i, i+x, i+1
            //         // only if those indices are within bounds
            //         if i + x_verts < total_verts && i + 1 < total_verts {
            //             indices.append(&mut vec![i as u32, (i + x_verts) as u32, (i + 1) as u32]);
            //             // println!("x_idx {} z_idx {} | adding {} {} {}", x_idx, z_idx, i as u32, (i + x_verts) as u32, (i + 1) as u32);
            //         }
            //         if i + 1 < total_verts && i - x_verts >= 0 {
            //             indices.append(&mut vec![i as u32, (i + 1) as u32, (i - x_verts) as u32]);
            //         }
            //     } else {
            //         if i + x_verts + 1 < total_verts && i + 1 < total_verts {
            //             indices.append(&mut vec![
            //                 i as u32,
            //                 (i + x_verts + 1) as u32,
            //                 (i + 1) as u32,
            //             ]);
            //         }
            //         if i + 1 < total_verts && i - x_verts + 1 >= 0 {
            //             indices.append(&mut vec![
            //                 i as u32,
            //                 (i + 1) as u32,
            //                 (i - x_verts + 1) as u32,
            //             ]);
            //         }
            //     }
            // }
        }
    }

    // print!("indices ");
    // for idx in indices.clone() {
    //     print!("{} ", idx)
    // }
    // println!();
    // let indices = triangulate(&delauney_verts).triangles.iter().map(|i| *i as u32).collect();
    return (verts, indices, colors);
}

pub fn x_marginal_cdf(x: f64) -> f64 {
    let m_z = Z_VIEW_DISTANCE as f64 * TILE_SIZE;
    let ptsqrt = PROB_TIGHTNESS.sqrt();
    return m_z * (x / (m_z.powi(2) + PROB_TIGHTNESS).sqrt()).asinh()
        + x * (m_z / (x.powi(2) + PROB_TIGHTNESS).sqrt()).asinh()
        - ptsqrt * (m_z * x / (ptsqrt * (x.powi(2) + m_z.powi(2) + m_z).sqrt())).atan();
}

pub fn inverse_x_marginal_cdf(y: f64) -> f64 {
    let mut x = y;
    for _ in 0..20 {
        let f_x = x_marginal_cdf(x);
        if y == f_x || (f_x - y).abs() / x.abs() <= 1e-7 {
            return x;
        }
        x = x * y / f_x;
    }
    // let f_x = x_marginal_cdf(x);
    // println!("EXCEEDED MAX ITER: {y} {x} {} ERROR: {}%", f_x, 100.*(f_x - y).abs() / x.abs());
    return x;
}

/// I'm SO SORRY for bad naming but y is actually z
/// Using y here because it's the canonical second variable for PDF's
pub fn y_given_x_cdf(y: f64, x: f64) -> f64 {
    let m_z = Z_VIEW_DISTANCE as f64 * TILE_SIZE;
    let xp1sqrt = (x.powi(2) + PROB_TIGHTNESS).sqrt();
    return (y / xp1sqrt).asinh() / (2. * (m_z / xp1sqrt).asinh());
}

pub fn inverse_y_given_x_cdf(z: f64, x: f64) -> f64 {
    let m_z = Z_VIEW_DISTANCE as f64 * TILE_SIZE;
    let xp1sqrt = (x.powi(2) + PROB_TIGHTNESS).sqrt();
    return xp1sqrt * (z * 2. * (m_z / xp1sqrt).asinh()).sinh();
}

/// Here we sample assuming the probability dist
/// p = 1/sqrt(x^2+1)*1/sqrt(z^2+1)
/// the cdf of p_x, c_x = asinh(x)
/// the inverse cdf is then sinh(x)
/// This makes our math look quite simple
/// Our points are in the range -x, x but we need to sample from -asinh(x * 1000), asinh(x * 1000)
/// -asinh(x * 1000), asinh(x * 1000) is because we are using an unnormalized PDF/CDF
/// Since the x and z dists are independent we can sample them independently
pub fn transform_lattice_positions(lattice: &mut Vec<DVec3>) {
    let m_z = Z_VIEW_DISTANCE as f64 * TILE_SIZE;
    let m_x = X_VIEW_DISTANCE as f64 * TILE_SIZE;
    for lattice_pos in lattice {
        let r = (lattice_pos.x.powi(2) + lattice_pos.z.powi(2)).sqrt();
        let theta = lattice_pos.z.atan2(lattice_pos.x);
        lattice_pos.x = r.sinh() * theta.cos();
        lattice_pos.z = r.sinh() * theta.sin();

        // first we get the x pos
        // let prev = lattice_pos.x;
        // println!(
        //     "X ({}): lin map [0, {}] -> [0, {}]: {} -> {} | inverse cdf map: {} -> {} | difference {}",
        //     lattice_pos.x,
        //     m_x,
        //     x_marginal_cdf(m_x),
        //     lattice_pos.x,
        //     lin_map(0., m_x, 0., x_marginal_cdf(m_x), lattice_pos.x),
        //     lattice_pos.x,
        //     inverse_x_marginal_cdf(lin_map(0., m_x, 0., x_marginal_cdf(m_x), lattice_pos.x)),
        //     lattice_pos.x - inverse_x_marginal_cdf(lin_map(0., m_x, 0., x_marginal_cdf(m_x), lattice_pos.x))
        // );
        // lattice_pos.x =
        //     inverse_x_marginal_cdf(lin_map(0., m_x, 0., x_marginal_cdf(m_x), thread_rng().gen_range(-m_x..m_x)));
        // let prev = lattice_pos.x;
        // println!("Actual diff: {}", prev - lattice_pos.x);

        // if lattice_pos.x.abs() < 1e-2 {
        //     println!(
        //         "Z ({})|X ({}) : lin map [0, {}] -> [0, {}]: {} -> {} | inverse cdf map: {} -> {} | difference {}",
        //         lattice_pos.z,
        //         lattice_pos.x,
        //         m_z,
        //         y_given_x_cdf(m_z, lattice_pos.x),
        //         lattice_pos.z,
        //         lin_map(
        //             0.,
        //             m_x,
        //             0.,
        //             y_given_x_cdf(m_z, lattice_pos.x),
        //             lattice_pos.z
        //         ),
        //         lattice_pos.z,
        //         inverse_y_given_x_cdf(
        //             lin_map(
        //                 0.,
        //                 m_z,
        //                 0.,
        //                 y_given_x_cdf(m_z, lattice_pos.x),
        //                 lattice_pos.z,
        //             ),
        //             lattice_pos.x,
        //         ),
        //         lattice_pos.z - inverse_y_given_x_cdf(
        //             lin_map(
        //                 0.,
        //                 m_z,
        //                 0.,
        //                 y_given_x_cdf(m_z, lattice_pos.x),
        //                 lattice_pos.z,
        //             ),
        //             lattice_pos.x,
        //         )
        //     );
        // }

        // lattice_pos.z = inverse_y_given_x_cdf(
        //     lin_map(
        //         0.,
        //         m_z,
        //         0.,
        //         y_given_x_cdf(m_z, prev),
        //         thread_rng().gen_range(-m_z..m_z),
        //     ),
        //     prev,
        // );
    }
}

pub fn compose_terrain_mesh(
    lattice: Vec<DVec3>,
    indices: Vec<u32>,
    colors: Vec<Vec4>,
    generator: &TerrainGenerator,
) -> Mesh {
    let mut tile_mesh = Mesh::new(PrimitiveTopology::TriangleList);
    let mut vertices: CompleteVertices = vec![];
    let mut delauney_verts: Vec<Point> = vec![];

    for vertex in lattice {
        let u = lin_map(-TILE_SIZE, TILE_SIZE, 0.0, 1.0, vertex.x) as f32;
        let v = lin_map(-TILE_SIZE, TILE_SIZE, 0.0, 1.0, vertex.z) as f32;
        let mut new_vec = vertex.as_vec3().to_array();
        let height = generator.get_height_for(vertex.x, vertex.z);
        // let height = 0.0; //(vertex.x*6.0 + vertex.z)*1.0/10.0;
        // let normal = Vec3::new(0., 1., 0.).to_array();
        let normal = generator
            .get_normal(vertex.x, vertex.z)
            .as_vec3()
            .to_array();
        new_vec[1] = height as f32;
        vertices.push((new_vec, normal, [u, v]));
        delauney_verts.push(Point {
            x: vertex.x,
            y: vertex.z,
        });
    }
    let positions: Vec<_> = vertices.iter().map(|(p, _, _)| *p).collect();
    let normals: Vec<_> = vertices.iter().map(|(_, n, _)| *n).collect();
    let uvs: Vec<_> = vertices.iter().map(|(_, _, uv)| *uv).collect();
    tile_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    tile_mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    tile_mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    // tile_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    let indices = triangulate(&delauney_verts)
        .triangles
        .iter()
        .map(|i| *i as u32)
        .collect();
    tile_mesh.set_indices(Some(Indices::U32(indices)));

    return tile_mesh;
}

/// Create a terrain mesh from the generator
pub fn create_terrain_mesh(generator: &TerrainGenerator) -> Mesh {
    let (mut verts, indices, colors) = create_lattice_plane();
    transform_lattice_positions(&mut verts);
    return compose_terrain_mesh(verts, indices, colors, &generator);
}
