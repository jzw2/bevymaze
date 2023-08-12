use crate::maze_gen::{CircleMaze, CircleNode, SquareMaze, SquareNode};
use bevy::math::{Vec2, Vec3};
use bevy::prelude::{shape, Mesh};
use bevy::render::mesh::{Indices, PrimitiveTopology};
use petgraph::graphmap::NodeTrait;
use std::f64::consts::PI;
use crate::render::{quad_cc_indices, quad_cw_indices, SimpleVertices};

/// A segment with endpoints p1 and p2
pub struct Segment {
    pub(crate) p1: (f64, f64),
    pub(crate) p2: (f64, f64),
}

/// A circle with center and radius
pub struct Circle {
    pub(crate) center: (f64, f64),
    pub(crate) radius: f64,
}

/// An arc of a circle with starting and ending angles a1 and a2
/// Should connect the angles counterclockwise
pub struct Arc {
    pub(crate) circle: Circle,
    pub(crate) a1: f64,
    pub(crate) a2: f64,
}

pub trait GetWall<N: NodeTrait> {
    /// Gets the wall
    fn get_wall_geometry(&self, width: f32, height: f32) -> Vec<Mesh>;

    /// Determine if a point is in a wall
    fn is_in_wall(&self, p: (f64, f64)) -> bool;
}

/// Convert a polar coordinate to it's cartesian counterpart
pub fn polar_to_cart(p: (f64, f64)) -> (f64, f64) {
    return (p.0 * p.1.cos(), p.0 * p.1.sin());
}

/// Get the distance between two points
pub fn dist(p1: (f64, f64), p2: (f64, f64)) -> f64 {
    return origin_dist((p1.0 - p2.0, p1.1 - p2.1));
}

/// Get the distance between a point and the origin
pub fn origin_dist(p: (f64, f64)) -> f64 {
    return (p.0 * p.0 + p.1 * p.1).sqrt();
}

/// Get the angle that the segment between the origin and p makes with the x-axis
/// Return value is in range from [0, 2pi)
pub fn polar_angle(p: (f64, f64)) -> f64 {
    return (p.1.atan2(p.0) + (2.0 * PI)).rem_euclid(2.0 * PI);
}

/// Convert a cartesian coordinate to polar
pub fn cart_to_polar(p: (f64, f64)) -> (f64, f64) {
    return (origin_dist(p), polar_angle(p));
}

/// Get the closest distance from a point to any point on a circle
pub fn distance_to_circle(circle: &Circle, p: (f64, f64)) -> f64 {
    let center_dist = dist(p, circle.center);
    return (circle.radius - center_dist).abs();
}

/// Get the closest distance from a point to any point on a segment
pub fn distance_to_segment(segment: &Segment, p: (f64, f64)) -> f64 {
    let v = (segment.p2.0 - segment.p1.0, segment.p2.1 - segment.p1.1);
    let p_off = (p.0 - segment.p1.0, p.1 - segment.p1.1);
    let t = (v.0 * p_off.0 + v.1 * p_off.1) / (v.0 * v.0 + v.1 * v.1);
    // clamp the projection
    let t = 0.0f64.max(1.0f64.min(t));
    // now get the dist
    let seg_p = (
        p.0 - (t * v.0 + segment.p1.0),
        p.1 - (t * v.1 + segment.p1.1),
    );
    return (seg_p.0 * seg_p.0 + seg_p.1 * seg_p.1).sqrt();
}

/// Get the mesh of a segment. The corners are listed clockwise.
pub fn get_segment_mesh(segment: &Segment, width: f32, height: f32) -> Mesh {
    // get the vector perpendicular to <p2-p1>
    let vp1 = Vec2::from((segment.p1.0 as f32, segment.p1.1 as f32));
    let vp2 = Vec2::from((segment.p2.0 as f32, segment.p2.1 as f32));
    let vp = vp2 - vp1;
    let vp_len = vp.length();

    // clockwise normal
    let normal = Vec2::from((vp.y, -vp.x)).normalize();
    let n_norm = -normal;

    let vp_unit = vp / vp_len;
    let face_vp_unit = [vp_unit.x, 0.0, vp_unit.y];
    let n_face_vp_unit = [-vp_unit.x, 0.0, -vp_unit.y];
    let face_norm = [normal.x, 0.0, normal.y];
    let n_face_norm = [-normal.x, 0.0, -normal.y];

    // true width
    let t_width = 2. * width;

    let v = vec![
        // bottom
        [(n_norm * width + vp1).x, 0.0, (n_norm * width + vp1).y],
        [(normal * width + vp1).x, 0.0, (normal * width + vp1).y],
        [(n_norm * width + vp2).x, 0.0, (n_norm * width + vp2).y],
        [(normal * width + vp2).x, 0.0, (normal * width + vp2).y],
        // top
        [(n_norm * width + vp1).x, height, (n_norm * width + vp1).y],
        [(normal * width + vp1).x, height, (normal * width + vp1).y],
        [(n_norm * width + vp2).x, height, (n_norm * width + vp2).y],
        [(normal * width + vp2).x, height, (normal * width + vp2).y],
    ];

    // the vertices with their normals and UV
    let mut vertices: Vec<([f32; 3], [f32; 3], [f32; 2])> = vec![];
    // the index positions; we update this every time we add a new quad
    let mut indices: Vec<u32> = vec![];
    let mut cur_idx_set = 0u32;

    // bottom
    vertices.append(&mut vec![
        (v[0], [0., -1., 0.], [0., 0.]),
        (v[1], [0., -1., 0.], [1., 0.]),
        (v[2], [0., -1., 0.], [0., vp_len/t_width]),
        (v[3], [0., -1., 0.], [1., vp_len/t_width]),
    ]);
    indices.append(&mut quad_cw_indices(cur_idx_set));
    cur_idx_set += 1;

    // top
    vertices.append(&mut vec![
        (v[4], [0., 1., 0.], [0., 0.]),
        (v[5], [0., 1., 0.], [1., 0.]),
        (v[6], [0., 1., 0.], [0., vp_len/t_width]),
        (v[7], [0., 1., 0.], [1., vp_len/t_width]),
    ]);
    indices.append(&mut quad_cc_indices(cur_idx_set));
    cur_idx_set += 1;

    // front
    vertices.append(&mut vec![
        (v[0], n_face_norm, [0., 0.]),
        (v[4], n_face_norm, [0., height/t_width]),
        (v[2], n_face_norm, [vp_len/t_width, 0.]),
        (v[6], n_face_norm, [vp_len/t_width, height/t_width]),
    ]);
    indices.append(&mut quad_cc_indices(cur_idx_set));
    cur_idx_set += 1;

    // back
    vertices.append(&mut vec![
        (v[1], face_norm, [0., 0.]),
        (v[5], face_norm, [0., height/t_width]),
        (v[3], face_norm, [vp_len/t_width, 0.]),
        (v[7], face_norm, [vp_len/t_width, height/t_width]),
    ]);
    indices.append(&mut quad_cw_indices(cur_idx_set));
    cur_idx_set += 1;

    // left
    vertices.append(&mut vec![
        (v[0], n_face_vp_unit, [0., 0.]),
        (v[1], n_face_vp_unit, [1., 0.]),
        (v[4], n_face_vp_unit, [0., height/t_width]),
        (v[5], n_face_vp_unit, [1.0, height/t_width]),
    ]);
    indices.append(&mut quad_cc_indices(cur_idx_set));
    cur_idx_set += 1;

    // right
    vertices.append(&mut vec![
        (v[2], face_vp_unit, [0., 0.]),
        (v[3], face_vp_unit, [1., 0.]),
        (v[6], face_vp_unit, [0., height/t_width]),
        (v[7], face_vp_unit, [1., height/t_width]),
    ]);
    indices.append(&mut quad_cw_indices(cur_idx_set));
    cur_idx_set += 1;

    let positions: Vec<_> = vertices.iter().map(|(p, _, _)| *p).collect();
    let normals: Vec<_> = vertices.iter().map(|(_, n, _)| *n).collect();
    let uvs: Vec<_> = vertices.iter().map(|(_, _, uv)| *uv).collect();

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.set_indices(Some(Indices::U32(indices)));
    return mesh;
}

/// get the closest distance from a point to any point on an arc
pub fn distance_to_arc(arc: &Arc, p: (f64, f64)) -> f64 {
    let p = (p.0 - arc.circle.center.0, p.1 - arc.circle.center.1);
    let p_angle = polar_angle(p);
    if arc.a1 > arc.a2 {
        if p_angle > arc.a1 || p_angle < arc.a2 {
            return distance_to_circle(&arc.circle, p);
        }
    } else {
        if arc.a1 < p_angle && p_angle < arc.a2 {
            return distance_to_circle(&arc.circle, p);
        }
    }
    let a1_p = polar_to_cart((arc.circle.radius, arc.a1));
    let a2_p = polar_to_cart((arc.circle.radius, arc.a2));

    return dist(a1_p, p).min(dist(a2_p, p));
}

/// Get an "extrusion mesh" of an arc, with the curve on the x-z plane
/// `divisions` is how many points between `a1` and `a2` to include
pub fn get_arc_mesh(arc: &Arc, width: f32, height: f32, divisions: u32) -> Mesh {
    let divisions = divisions + 2;

    // simple vertex positions
    let mut v: SimpleVertices = Vec::new();

    // first construct the list of points that make up the bottom of the arc mesh
    for i in 0..divisions {
        // we are parametrically walking around the arc, sampling points
        let t = (i as f64) / (divisions as f64 - 1.0);
        // inner points are the points closest to the center
        let inner_point = polar_to_cart((
            arc.circle.radius - width as f64,
            (arc.a2 - arc.a1) * t + arc.a1,
        ));
        // outer points are the points furthest from the center
        let outer_point = polar_to_cart((
            arc.circle.radius + width as f64,
            (arc.a2 - arc.a1) * t + arc.a1,
        ));
        v.push([
            (inner_point.0 + arc.circle.center.0) as f32,
            0.0,
            (inner_point.1 + arc.circle.center.1) as f32,
        ]);
        v.push([
            (outer_point.0 + arc.circle.center.0) as f32,
            0.0,
            (outer_point.1 + arc.circle.center.1) as f32,
        ]);
    }
    // maybe someone smarter can compress this into one loop, but that person is not me
    for i in 0..divisions {
        // we are parametrically walking around the arc, sampling points
        let t = (i as f64) / (divisions as f64 - 1.0);
        // inner points are the points closest to the center
        let inner_point = polar_to_cart((
            arc.circle.radius - width as f64,
            (arc.a2 - arc.a1) * t + arc.a1,
        ));
        // outer points are the points furthest from the center
        let outer_point = polar_to_cart((
            arc.circle.radius + width as f64,
            (arc.a2 - arc.a1) * t + arc.a1,
        ));
        v.push([
            (inner_point.0 + arc.circle.center.0) as f32,
            height,
            (inner_point.1 + arc.circle.center.1) as f32,
        ]);
        v.push([
            (outer_point.0 + arc.circle.center.0) as f32,
            height,
            (outer_point.1 + arc.circle.center.1) as f32,
        ]);
    }

    // the vertices with their normals and UV
    let mut vertices: Vec<([f32; 3], [f32; 3], [f32; 2])> = vec![];
    // the index positions; we update this every time we add a new quad
    let mut indices: Vec<u32> = vec![];
    let mut cur_idx_set = 0u32;

    // create the bottom
    let b_norm = [0.0f32, -1.0, 0.0];
    for i in 0..divisions - 1 {
        let i0 = (i * 2) as usize;
        let i1 = i0 + 1;
        let i2 = i0 + 2;
        let i3 = i0 + 3;
        vertices.append(&mut vec![
            (v[i0], b_norm, [0.0, 0.]),
            (v[i1], b_norm, [1.0, 0.]),
            (v[i2], b_norm, [0.0, 1.]),
            (v[i3], b_norm, [1.0, 1.]),
        ]);
        // add verts clockwise
        indices.append(&mut quad_cw_indices(cur_idx_set));
        cur_idx_set += 1;
    }
    // create the top
    let t_norm = [0.0f32, 1.0, 0.0];
    for i in 0..divisions - 1 {
        let i0 = (i * 2 + divisions * 2) as usize;
        let i1 = i0 + 1;
        let i2 = i0 + 2;
        let i3 = i0 + 3;
        vertices.append(&mut vec![
            (v[i0], t_norm, [0.0, 0.]),
            (v[i1], t_norm, [1.0, 0.]),
            (v[i2], t_norm, [0.0, 1.]),
            (v[i3], t_norm, [1.0, 1.]),
        ]);
        indices.append(&mut quad_cc_indices(cur_idx_set));
        cur_idx_set += 1;
    }

    // inner
    for i in 0..divisions - 1 {
        let i0 = (i * 2) as usize;
        let i1 = i0 + 2;
        let i2 = i0 + 2 * divisions as usize;
        let i3 = i2 + 2;

        let i0_norm = (Vec3::from(v[i0]) - Vec3::from(v[i0 + 1])).normalize();
        let i0_norm = [i0_norm.x, i0_norm.y, i0_norm.z];

        let i1_norm = (Vec3::from(v[i1]) - Vec3::from(v[i1 + 1])).normalize();
        let i1_norm = [i1_norm.x, i1_norm.y, i1_norm.z];

        vertices.append(&mut vec![
            (v[i0], i0_norm, [0.0, 0.]),
            (v[i1], i0_norm, [1.0, 0.]),
            (v[i2], i1_norm, [0.0, 1.]),
            (v[i3], i1_norm, [1.0, 1.]),
        ]);
        indices.append(&mut quad_cw_indices(cur_idx_set));
        cur_idx_set += 1;
    }

    // outer
    for i in 0..divisions - 1 {
        let i0 = (i * 2) as usize + 1;
        let i1 = i0 + 2;
        let i2 = i0 + 2 * divisions as usize;
        let i3 = i2 + 2;

        let i0_norm = (Vec3::from(v[i0]) - Vec3::from(v[i0 - 1])).normalize();
        let i0_norm = [i0_norm.x, i0_norm.y, i0_norm.z];

        let i1_norm = (Vec3::from(v[i1]) - Vec3::from(v[i1 - 1])).normalize();
        let i1_norm = [i1_norm.x, i1_norm.y, i1_norm.z];

        vertices.append(&mut vec![
            (v[i0], i0_norm, [0.0, 0.]),
            (v[i1], i0_norm, [1.0, 0.]),
            (v[i2], i1_norm, [0.0, 1.]),
            (v[i3], i1_norm, [1.0, 1.]),
        ]);
        indices.append(&mut quad_cc_indices(cur_idx_set));
        cur_idx_set += 1;
    }

    // clockwise end
    let i0 = 0usize;
    let i1 = 1usize;
    let i2 = 2 * divisions as usize;
    let i3 = i2 + 1;
    // clockwise normal to the inner normal
    // should point clockwise along the circle after rotation
    let c_norm = (Vec3::from(v[i0]) - Vec3::from(v[i0 + 1])).normalize();
    let c_norm = [c_norm.z, c_norm.y, -c_norm.x];
    vertices.append(&mut vec![
        (v[i0], c_norm, [0.0, 0.]),
        (v[i1], c_norm, [1.0, 0.]),
        (v[i2], c_norm, [0.0, 1.]),
        (v[i3], c_norm, [1.0, 1.]),
    ]);
    indices.append(&mut quad_cc_indices(cur_idx_set));
    cur_idx_set += 1;

    // counterclockwise end
    let i0 = (2 * divisions - 2) as usize;
    let i1 = i0 + 1;
    let i2 = (2 * 2 * divisions - 2) as usize;
    let i3 = i2 + 1;
    // clockwise normal to the outer normal
    // should point counterclockwise along the circle after the clockwise rotation (confusingly)
    let cc_norm = (Vec3::from(v[i0 + 1]) - Vec3::from(v[i0])).normalize();
    let cc_norm = [cc_norm.z, cc_norm.y, -cc_norm.x];
    vertices.append(&mut vec![
        (v[i0], cc_norm, [0.0, 0.]),
        (v[i1], cc_norm, [1.0, 0.]),
        (v[i2], cc_norm, [0.0, 1.]),
        (v[i3], cc_norm, [1.0, 1.]),
    ]);
    indices.append(&mut quad_cw_indices(cur_idx_set));
    cur_idx_set += 1;

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

impl CircleMaze {
    fn get_node_cart_point(&self, n: CircleNode) -> (f64, f64) {
        return polar_to_cart(self.get_node_pol_point(n));
    }

    fn get_node_pol_point(&self, n: CircleNode) -> (f64, f64) {
        return (
            n.0 as f64 * self.cell_size,
            n.1 as f64 / (self.nodes_at_radius(n.0) as f64) * 2.0 * PI,
        );
    }
}

impl CircleMaze {
    fn get_node_meshes(&self, node: CircleNode, width: f32, height: f32) -> Vec<Mesh> {
        let mut meshes: Vec<Mesh> = vec![];

        for touching in self.touching(node, 0.0) {
            if touching.0 == node.0 && touching.1 > node.1 {
                let mut clockwise_wall_node = touching;
                if touching.1 < node.1 {
                    // if the point's closest node is farther along the circle,
                    // then we are using the clockwise wall
                    // default is to use the counterclockwise wall
                    clockwise_wall_node = node;
                }
                let node_pol_point = self.get_node_pol_point(clockwise_wall_node);
                if !self
                    .maze
                    .contains_edge(self.correct_node(node), self.correct_node(touching))
                {
                    meshes.push(get_segment_mesh(
                        &Segment {
                            p1: polar_to_cart(node_pol_point),
                            p2: polar_to_cart((
                                node_pol_point.0 + self.cell_size,
                                node_pol_point.1,
                            )),
                        },
                        width,
                        height,
                    ));
                }
            } else if touching.0 > node.0 {
                let (npp_radius, npp_theta) = self.get_node_pol_point(node);
                let (_, npp_theta_plus_1) = self.get_node_pol_point((node.0, node.1 + 1));
                let (tpp_radius, tpp_theta) = self.get_node_pol_point(touching);
                let (_, tpp_theta_plus_1) = self.get_node_pol_point((touching.0, touching.1 + 1));

                let mut thetas = vec![npp_theta, npp_theta_plus_1, tpp_theta, tpp_theta_plus_1];

                //pro gamer avoid dividing my zero
                thetas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                if !self
                    .maze
                    .contains_edge(self.correct_node(node), self.correct_node(touching))
                {
                    meshes.push(get_arc_mesh(
                        &Arc {
                            circle: Circle {
                                center: (0.0, 0.0),
                                radius: npp_radius.max(tpp_radius),
                            },
                            a1: thetas[1],
                            a2: thetas[2],
                        },
                        width,
                        height,
                        3,
                    ));
                }
            }
        }
        return meshes;
    }
}

impl GetWall<SquareNode> for SquareMaze {
    fn get_wall_geometry(&self, width: f32, height: f32) -> Vec<Mesh> {
        let mut meshes: Vec<Mesh> = vec![];
        for x in 0..self.size {
            for y in 0..self.size {
                let cur = (x + self.offset.0, y + self.offset.1);
                let corner = (x + self.offset.0 + 1, y + self.offset.1 + 1);
                let adjacent = (x + self.offset.0 + 1, y + self.offset.1);
                if !self.maze.contains_edge(cur, adjacent) {
                    meshes.push(get_segment_mesh(&Segment {
                        p1: (adjacent.0 as f64, adjacent.1 as f64),
                        p2: (corner.0 as f64, corner.1 as f64)
                    }, width, height));
                }
                let adjacent = (x + self.offset.0, y + self.offset.1 + 1);
                if !self.maze.contains_edge(cur, adjacent) {
                    meshes.push(get_segment_mesh(&Segment {
                        p1: (adjacent.0 as f64, adjacent.1 as f64),
                        p2: (corner.0 as f64, corner.1 as f64)
                    }, width, height));
                }
            }
        }
        return meshes;
    }

    fn is_in_wall(&self, p: (f64, f64)) -> bool {
        todo!()
    }
}

impl GetWall<CircleNode> for CircleMaze {
    fn get_wall_geometry(&self, width: f32, height: f32) -> Vec<Mesh> {
        let mut meshes: Vec<Mesh> = vec![];
        for r in 0..self.radius + 1 {
            for n in 0..self.nodes_at_radius(r) {
                meshes.append(&mut self.get_node_meshes((r, n as i64), width, height));
            }
        }
        return meshes;
    }

    fn is_in_wall(&self, p: (f64, f64)) -> bool {
        // get the closest sector to this point
        let pp = cart_to_polar(p);
        let r = (pp.0 / self.cell_size).floor() as u64;
        if r > self.radius {
            return false;
        }
        let node_count = self.nodes_at_radius(r);
        let t = (pp.1 / (2.0 * PI / (node_count as f64))).floor() as i64;
        let n = (r, t);

        for touching in self.touching(n, 0.0) {
            if touching.0 == n.0 {
                let mut clockwise_wall_node = touching;
                if touching.1 < n.1 {
                    // if the point's closest node is farther along the circle,
                    // then we are using the clockwise wall
                    // default is to use the counterclockwise wall
                    clockwise_wall_node = n;
                }
                let node_pol_point = self.get_node_pol_point(clockwise_wall_node);
                if distance_to_segment(
                    &Segment {
                        p1: polar_to_cart(node_pol_point),
                        p2: polar_to_cart((node_pol_point.0 + self.cell_size, node_pol_point.1)),
                    },
                    p,
                ) <= self.wall_width
                    && !self
                        .maze
                        .contains_edge(self.correct_node(n), self.correct_node(touching))
                {
                    return true;
                }
            } else {
                let (npp_radius, npp_theta) = self.get_node_pol_point(n);
                let (_, npp_theta_plus_1) = self.get_node_pol_point((n.0, n.1 + 1));
                let (tpp_radius, tpp_theta) = self.get_node_pol_point(touching);
                let (_, tpp_theta_plus_1) = self.get_node_pol_point((touching.0, touching.1 + 1));

                let mut thetas = vec![npp_theta, npp_theta_plus_1, tpp_theta, tpp_theta_plus_1];

                //pro gamer avoid dividing my zero
                thetas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let arc: Arc = Arc {
                    circle: Circle {
                        center: (0.0, 0.0),
                        radius: npp_radius.max(tpp_radius),
                    },
                    a1: thetas[1],
                    a2: thetas[2],
                };

                if distance_to_arc(&arc, p) <= self.wall_width
                    && !self
                        .maze
                        .contains_edge(self.correct_node(n), self.correct_node(touching))
                {
                    return true;
                }
            }
        }

        return false;
    }
}
