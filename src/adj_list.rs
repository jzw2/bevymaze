use crate::terrain_render::{X_VIEW_DIST_M, Z_VIEW_DIST_M};
use delaunator::{triangulate, Point};
use kiddo::{KdTree, SquaredEuclidean};
use rand::{thread_rng, Rng};
use server::util::lin_map;

type TPoint2d = [f64; 2];

type Triangle2d = (TPoint2d, TPoint2d, TPoint2d);

fn get_barycentric_coordinates(point: TPoint2d, triangle: Triangle2d) -> (f64, f64, f64) {
    let p0 = triangle.0;
    let p1 = triangle.1;
    let p2 = triangle.2;
    let area =
        0.5 * (-p1[1] * p2[0] + p0[1] * (-p1[0] + p2[0]) + p0[0] * (p1[1] - p2[1]) + p1[0] * p2[1]);
    let s = 1. / (2. * area)
        * (p0[1] * p2[0] - p0[0] * p2[1] + (p2[1] - p0[1]) * point[0] + (p0[0] - p2[0]) * point[1]);
    let t = 1. / (2. * area)
        * (p0[0] * p1[1] - p0[1] * p1[0] + (p0[1] - p1[1]) * p0[0] + (p1[0] - p0[0]) * point[1]);
    return (s, t, 1. - s - t);
}

/// A (triangle index, from edge angle) pair
type TriangleAngle2d = (usize, f64);

struct AdjacencyList {
    /// Each entry in the adjacency list is a list of triangles that have OUR vertex
    /// as one of said triangles vertices
    /// The angle is the angle of the edge in said triangle that goes FROM OUR vertex outward
    adj_list: Vec<Vec<TriangleAngle2d>>,
    /// The list of initial vertices without any extra info
    vertices: Vec<TPoint2d>,
    /// The triangulation gotten from delaunator;
    /// all the triangles' vertices are listed counterclockwise
    triangles: Vec<usize>,
    /// A 2D KD-tree of all the vertices
    /// The "items" attached to each vertex are the indices of the vertex as seen in `vertices`
    tree: KdTree<f64, 2>,
}

impl AdjacencyList {
    /// Builds the adjacency list by
    /// 1. constructing the delaunay triangulation
    /// 2. finding all the triangles that belong to the vert
    /// 3. sorting the triangles by the angle of the edge that goes FROM the vertex in question
    pub fn build(verts: Vec<TPoint2d>) -> Self {
        let delaunay_points: Vec<Point> = verts
            .iter()
            .map(|v| Point {
                x: (*v)[0],
                y: (*v)[1],
            })
            .collect();
        let triangulation = triangulate(&delaunay_points);
        let mut adj_list: Vec<Vec<TriangleAngle2d>> = vec![vec![]; verts.len()];
        for (vert_in_triangle_idx, vert_idx) in triangulation.triangles.iter().enumerate() {
            // the triangulation consists of a list of 3-tuples of vertex INDICES
            // we will use that vertex index to index into our adjacency list
            let triangle_idx = vert_in_triangle_idx / 3;

            // get the vector representing the edge FROM the vertex we found
            // this is the vertex we sort the triangle by
            let edge_beg = verts[*vert_idx];
            let triangle_start = triangle_idx * 3;
            let cur_idx = vert_in_triangle_idx - triangle_start;
            let next_idx = (cur_idx + 1) % 3;
            let actual_next_idx = next_idx + triangle_start;
            let edge_end = verts[triangulation.triangles[actual_next_idx]];
            let vec_from_us = [edge_end[0] - edge_beg[0], edge_end[1] - edge_beg[1]];
            let angle = vec_from_us[1].atan2(vec_from_us[0]);

            let new_pos = adj_list[*vert_idx]
                .binary_search_by(|tri| tri.1.partial_cmp(&angle).unwrap())
                .unwrap_or_else(|p| p);
            adj_list[*vert_idx].insert(new_pos, (triangle_idx, angle));
        }

        return AdjacencyList {
            adj_list,
            tree: KdTree::<f64, 2>::from(&verts),
            vertices: verts,
            triangles: triangulation.triangles,
        };
    }

    /// Find the triangle in our adjacency list
    /// The process is:
    pub fn get_tri(&self, point: TPoint2d) -> Option<usize> {
        let nearest = self.tree.nearest_one::<SquaredEuclidean>(&point);
        let to_search = &self.adj_list[nearest.item as usize];
        // we use the angle of ourselves for the binary search
        let angle = point[1].atan2(point[0]);
        for tri in to_search {
            let p0 = self.vertices[self.triangles[tri.0 * 3]];
            let p1 = self.vertices[self.triangles[tri.0 * 3 + 1]];
            let p2 = self.vertices[self.triangles[tri.0 * 3 + 2]];
            let area = 0.5
                * (-p1[1] * p2[0]
                    + p0[1] * (-p1[0] + p2[0])
                    + p0[0] * (p1[1] - p2[1])
                    + p1[0] * p2[1]);
            let s = 1. / (2. * area)
                * (p0[1] * p2[0] - p0[0] * p2[1]
                    + (p2[1] - p0[1]) * point[0]
                    + (p0[0] - p2[0]) * point[1]);
            let t = 1. / (2. * area)
                * (p0[0] * p1[1] - p0[1] * p1[0]
                    + (p0[1] - p1[1]) * p0[0]
                    + (p1[0] - p0[0]) * point[1]);
            if s >= 0. && t >= 0. && 1. - s - t >= 0. {
                return Some(tri.0);
            }
        }

        return None;

        let res = to_search.binary_search_by(|tri| tri.1.partial_cmp(&angle).unwrap());
        let tri_idx = match res {
            // we fall on the boundary of a triangle, so it's OK to use that idx.
            Ok(p) => to_search[p].0,
            // it's the first item greater than our angle; it's necessary to use the triangle
            // "before" the one found, because that's the triangle that OUR angle is just greater
            // than, and hence we may actually lie in. Apply the modulo because the triangle
            // boundaries can wrap around the entire vertex
            Err(p) => to_search[(p + 1) % to_search.len()].0,
        };
        // we still need to do a collision check. When the vertex is on the edge of the
        // triangulation, we can get erroneous results.
        // email pregracke@gmail.com for a full explanation

        // triangle intersection test adapted from https://stackoverflow.com/a/14382692/3210986
        // TODO: remove code duplication
        let p0 = self.vertices[self.triangles[tri_idx * 3]];
        let p1 = self.vertices[self.triangles[tri_idx * 3 + 1]];
        let p2 = self.vertices[self.triangles[tri_idx * 3 + 2]];
        let area = 0.5
            * (-p1[1] * p2[0] + p0[1] * (-p1[0] + p2[0]) + p0[0] * (p1[1] - p2[1]) + p1[0] * p2[1]);
        let s = 1. / (2. * area)
            * (p0[1] * p2[0] - p0[0] * p2[1]
                + (p2[1] - p0[1]) * point[0]
                + (p0[0] - p2[0]) * point[1]);
        let t = 1. / (2. * area)
            * (p0[0] * p1[1] - p0[1] * p1[0]
                + (p0[1] - p1[1]) * p0[0]
                + (p1[0] - p0[0]) * point[1]);
        if s >= 0. && t >= 0. && 1. - s - t >= 0. {
            return Some(tri_idx);
        }

        return None;
    }
}

#[test]
fn one_triangle_test() {
    let adj_list = AdjacencyList::build(vec![[0., 0.], [-1., 1.], [1., 1.]]);
    assert_eq!(adj_list.adj_list.len(), 3);
    assert_eq!(adj_list.adj_list[0].len(), 1);
    assert_eq!(adj_list.adj_list[1].len(), 1);
    assert_eq!(adj_list.adj_list[2].len(), 1);
    let found = adj_list.get_tri([0., 0.5]).unwrap();
    assert_eq!(found, 0);
}

#[test]
fn many_triangles_test() {
    let mut points = vec![];
    for _ in 0..100000 {
        let p = [
            thread_rng().gen_range(-40000.0..40000.0),
            thread_rng().gen_range(-300000.0..300000.0),
        ];
        points.push([p[0], p[1]]);
    }
    let adj_list = AdjacencyList::build(points);

    adj_list.get_tri([0., 0.]).unwrap();
}

#[test]
fn many_points_many_triangles_bench() {
    let mut points = vec![];
    for x in 0..100 {
        for y in 0..1000 {
            points.push([
                lin_map(0., 100., -X_VIEW_DIST_M, X_VIEW_DIST_M, x as f64)
                    + thread_rng().gen_range(-1.0..1.0),
                lin_map(0., 1000., -Z_VIEW_DIST_M, Z_VIEW_DIST_M, y as f64)
                    + thread_rng().gen_range(-1.0..1.0),
            ]);
        }
    }
    let adj_list = AdjacencyList::build(points);

    let mut nfc = 0;
    for _ in 0..100000 {
        let p = [
            thread_rng().gen_range(-X_VIEW_DIST_M..X_VIEW_DIST_M),
            thread_rng().gen_range(-Z_VIEW_DIST_M..Z_VIEW_DIST_M),
        ];
        if adj_list.get_tri(p).is_none() {
            nfc += 1;
        }
    }
    println!("{} not found", nfc);

    for _ in 0..100 {
        let p = [
            thread_rng().gen_range(-X_VIEW_DIST_M..X_VIEW_DIST_M),
            thread_rng().gen_range(-Z_VIEW_DIST_M..Z_VIEW_DIST_M),
        ];
        let nearest = adj_list.tree.nearest_one::<SquaredEuclidean>(&p);
        let vert = adj_list.vertices[nearest.item as usize];

        println!(
            "{} nearest to {} {} is {} {}",
            nearest.distance.sqrt(),
            p[0],
            p[1],
            vert[0],
            vert[1]
        )
    }
}
