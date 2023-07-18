use crate::maze_gen::{CircleMaze, CircleNode};
use bevy::math::{Vec2, Vec3};
use bevy::prelude::{shape, Mesh};
use bevy::render::mesh::{Indices, PrimitiveTopology};
use petgraph::graphmap::NodeTrait;
use std::f64::consts::PI;

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
    fn get_wall_geometry(&self, height: f64) -> Mesh;

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
    let normal = Vec2::from((vp.y, -vp.x)).normalize() * width;
    let n_norm = -normal;

    let vp_unit = vp.normalize();
    let face_vp_unit = [vp_unit.x, 0.0, vp_unit.y];
    let n_face_vp_unit = [-vp_unit.x, 0.0, -vp_unit.y];
    let face_norm = [normal.x, 0.0, normal.y];
    let n_face_norm = [-normal.x, 0.0, -normal.y];

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);

    let v = vec![
        // bottom
        [(normal + vp1).x, 0.0, (normal + vp1).y],
        [(n_norm + vp1).x, 0.0, (n_norm + vp1).y],
        [(normal + vp2).x, 0.0, (normal + vp2).y],
        [(n_norm + vp2).x, 0.0, (n_norm + vp2).y],
        // top
        [(normal + vp1).x, height, (normal + vp1).y],
        [(n_norm + vp1).x, height, (n_norm + vp1).y],
        [(normal + vp2).x, height, (normal + vp2).y],
        [(n_norm + vp2).x, height, (n_norm + vp2).y],
    ];

    let complete_vertices = &[
        // Front
        (v[1], n_face_norm, [0., 0.]),
        (v[3], n_face_norm, [1.0, 0.]),
        (v[5], n_face_norm, [1.0, 1.0]),
        (v[7], n_face_norm, [0., 1.0]),
        // Back
        (v[0], face_norm, [1.0, 0.]),
        (v[2], face_norm, [0., 0.]),
        (v[4], face_norm, [0., 1.0]),
        (v[6], face_norm, [1.0, 1.0]),
        // Right
        (v[2], face_vp_unit, [0., 0.]),
        (v[3], face_vp_unit, [1.0, 0.]),
        (v[6], face_vp_unit, [1.0, 1.0]),
        (v[7], face_vp_unit, [0., 1.0]),
        // Left
        (v[0], n_face_vp_unit, [1.0, 0.]),
        (v[1], n_face_vp_unit, [0., 0.]),
        (v[4], n_face_vp_unit, [0., 1.0]),
        (v[5], n_face_vp_unit, [1.0, 1.0]),
        // Top
        (v[4], [0., 1.0, 0.], [1.0, 0.]),
        (v[6], [0., 1.0, 0.], [0., 0.]),
        (v[5], [0., 1.0, 0.], [0., 1.0]),
        (v[7], [0., 1.0, 0.], [1.0, 1.0]),
        // Bottom
        (v[0], [0., -1.0, 0.], [0., 0.]),
        (v[1], [0., -1.0, 0.], [1.0, 0.]),
        (v[2], [0., -1.0, 0.], [1.0, 1.0]),
        (v[3], [0., -1.0, 0.], [0., 1.0]),
    ];

    let positions: Vec<_> = complete_vertices.iter().map(|(p, _, _)| *p).collect();
    let normals: Vec<_> = complete_vertices.iter().map(|(_, n, _)| *n).collect();
    let uvs: Vec<_> = complete_vertices.iter().map(|(_, _, uv)| *uv).collect();

    let indices = Indices::U32(vec![
        0, 1, 2, 2, 3, 0, // front
        4, 5, 6, 6, 7, 4, // back
        8, 9, 10, 10, 11, 8, // right
        12, 13, 14, 14, 15, 12, // left
        16, 17, 18, 18, 19, 16, // top
        20, 21, 22, 22, 23, 20, // bottom
    ]);

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList);
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.set_indices(Some(indices));
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

impl GetWall<CircleNode> for CircleMaze {
    fn get_wall_geometry(&self, height: f64) -> Mesh {
        todo!()
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
