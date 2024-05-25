use crate::render::{tri_cc_indices, tri_cw_indices, CompleteVertices};
use bevy::math::{Quat, Vec2, Vec3};
use bevy::prelude::Mesh;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use rand::rngs::ThreadRng;
use rand::{thread_rng, Rng};
use std::f64::consts::PI;
use bevy::utils::default;

const TREE_MIN_HEIGHT: f64 = 10.;
const TREE_MAX_HEIGHT: f64 = 30.;
const TREE_MAX_EXTENSION: f64 = TREE_MAX_HEIGHT - TREE_MIN_HEIGHT;
const CONE_MIN_HEIGHT_PERC: f64 = 0.3;
const CONE_MAX_HEIGHT_PERC: f64 = 0.6;

fn cone_point_generator(rng: &mut ThreadRng, number: usize, radius: f64, height: f64) -> Vec<Vec3> {
    let mut points = vec![];
    let slope = height / radius;
    for i in 0..number {
        // generate our x/y coords
        let x_p = rng.gen_range(0.0..radius);
        let y = rng.gen_range(0.0..height);
        // generate a random rotation
        let theta = rng.gen_range(0.0..(2. * PI));
        // we want a point in the left slice of the cone cross-section,
        // which is a right-triangle
        // we can generate a point in the triangle's bounding box and flip the x and y
        // if it falls outside the triangle
        if y / x_p <= slope {
            points.push(Vec3::new(x_p as f32, y as f32, 0.));
        } else {
            points.push(Vec3::new((radius - x_p) as f32, (height - y) as f32, 0.));
        }
        // now shift the x coordinate back across the y axis
        // do this because the triangle we just generated the points in goes from
        // (0, 0) -> (radius, height) -> (radius, 0)
        // and we want points in the triangle that goes from
        // (0, 0) -> (0, height), -> (-radius, 0)
        points[i].x -= radius as f32;
        // rotate the x coordinate
        // our xP changes based on if the point is inside our outside the triangle
        // hence using points[0] instead of x
        let x = points[i].x as f64 * theta.cos();
        let z = points[i].x as f64 * theta.sin();
        points[i].x = x as f32;
        points[i].z = z as f32;
    }
    return points;
}

/// get an equilateral triangle around a point
/// with a random orientation
pub fn get_equilateral_tri_around(
    rng: &mut ThreadRng,
    norm: Vec3,
    point: Vec3,
    radius: f64,
) -> [Vec3; 3] {
    // all points on the plane defined by `point` and `norm`
    // satisfy (v - point) * norm = 0
    // aka (vx - px)nx + (vy - py)ny + (vz - pz)nz = 0
    // we can choose two of these (e.g. vx and vy) randomly and
    // then solve for vz with
    // pz - ( (vx - px)nx + (vy - py)ny ) / nz  = vz
    let top_dir_v1 = rng.gen_range(0.0..1.0);
    let top_dir_v2 = rng.gen_range(0.0..1.0);
    let top_dir: Vec3;
    if norm.z != 0. {
        top_dir = Vec3::new(
            top_dir_v1,
            top_dir_v2,
            point.z - ((top_dir_v1 - point.x) * norm.x + (top_dir_v2 - point.y) * norm.y) / norm.z,
        )
        .normalize();
    } else if norm.x != 0. {
        top_dir = Vec3::new(
            point.x - ((top_dir_v1 - point.y) * norm.y + (top_dir_v2 - point.z) * norm.z) / norm.x,
            top_dir_v1,
            top_dir_v2,
        )
        .normalize();
    } else if norm.y != 0. {
        top_dir = Vec3::new(
            top_dir_v1,
            point.y - ((top_dir_v1 - point.x) * norm.x + (top_dir_v2 - point.z) * norm.z) / norm.y,
            top_dir_v2,
        )
        .normalize();
    } else {
        panic!("BAD NORM")
    }

    let angle = (2. * PI / 3.) as f32;
    let p1 = point + top_dir * (radius as f32);
    let p2 = Quat::from_axis_angle(norm, angle).mul_vec3(p1 - point) + point;
    let p3 = Quat::from_axis_angle(norm, angle * 2.).mul_vec3(p1 - point) + point;
    return [p1, p2, p3];
}

pub fn get_tree_mesh(seed: i32) -> Mesh {
    // first get the trunk
    let mut rng = thread_rng();
    let height = TREE_MIN_HEIGHT + rng.gen_range(0.0..TREE_MAX_EXTENSION);
    let cone_height = height * rng.gen_range(CONE_MIN_HEIGHT_PERC..CONE_MAX_HEIGHT_PERC);
    let leaf_points = cone_point_generator(&mut rng, 20, cone_height / 2., cone_height);

    let mut vertices: CompleteVertices = vec![];
    let mut indices: Vec<u32> = vec![];
    let mut i = 0;
    for point in leaf_points {
        let norm = Vec3::new(
            rng.gen_range(0.0..1.0),
            rng.gen_range(0.0..1.0),
            rng.gen_range(0.0..1.0),
        )
        .normalize();
        let norm = -point.clone().normalize();
        let tri = get_equilateral_tri_around(&mut rng, norm, point, 5.);
        vertices.push((tri[0].to_array(), norm.to_array(), [0., 0.]));
        vertices.push((tri[1].to_array(), norm.to_array(), [1., 0.]));
        vertices.push((tri[2].to_array(), norm.to_array(), [0., 1.]));
        indices.append(&mut tri_cc_indices(i));
        i += 1;
    }

    // trunk
    let trunk_base = get_equilateral_tri_around(
        &mut rng,
        -Vec3::Y,
        Vec3::new(0., -(height - cone_height) as f32, 0.),
        5.,
    );
    let trunk_top = Vec3::new(0., cone_height as f32, 0.);
    // bottom
    vertices.push((trunk_base[0].to_array(), (-Vec3::Y).to_array(), [0., 0.]));
    vertices.push((trunk_base[1].to_array(), (-Vec3::Y).to_array(), [1., 0.]));
    vertices.push((trunk_base[2].to_array(), (-Vec3::Y).to_array(), [0., 1.]));
    indices.append(&mut tri_cw_indices(i));
    i += 1;

    // trunk side 1
    let side_1_norm = (trunk_base[1] - trunk_base[0])
        .cross(trunk_top - trunk_base[0])
        .normalize();
    vertices.push((trunk_base[0].to_array(), side_1_norm.to_array(), [0., 0.]));
    vertices.push((trunk_base[1].to_array(), side_1_norm.to_array(), [1., 0.]));
    vertices.push((trunk_top.to_array(), side_1_norm.to_array(), [0., 1.]));
    indices.append(&mut tri_cw_indices(i));
    i += 1;

    // trunk side 2
    let side_2_norm = (trunk_base[2] - trunk_base[1])
        .cross(trunk_top - trunk_base[1])
        .normalize();
    vertices.push((trunk_base[1].to_array(), side_2_norm.to_array(), [0., 0.]));
    vertices.push((trunk_base[2].to_array(), side_2_norm.to_array(), [1., 0.]));
    vertices.push((trunk_top.to_array(), side_2_norm.to_array(), [0., 1.]));
    indices.append(&mut tri_cw_indices(i));
    i += 1;

    // trunk side 3
    let side_3_norm = (trunk_base[0] - trunk_base[2])
        .cross(trunk_top - trunk_base[2])
        .normalize();
    vertices.push((trunk_base[2].to_array(), side_3_norm.to_array(), [0., 0.]));
    vertices.push((trunk_base[0].to_array(), side_3_norm.to_array(), [1., 0.]));
    vertices.push((trunk_top.to_array(), side_3_norm.to_array(), [0., 1.]));
    indices.append(&mut tri_cw_indices(i));
    i += 1;

    let positions: Vec<_> = vertices.iter().map(|(p, _, _)| *p).collect();
    let normals: Vec<_> = vertices.iter().map(|(_, n, _)| *n).collect();
    let uvs: Vec<_> = vertices.iter().map(|(_, _, uv)| *uv).collect();

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    // mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    return mesh;
}
