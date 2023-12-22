// The whole idea here is to create 3d voronoi cells,
// so that the cross section on the x-y plane matches the triangulation

use crate::terrain_render::{X_VIEW_DIST_M, Z_VIEW_DIST_M};
use bevy::render::render_resource::encase::private::RuntimeSizedArray;
use delaunator::{triangulate, Point, Triangulation, EMPTY};
use kiddo::KdTree;
use rand::{thread_rng, Rng};
use server::util::{dist, lin_map};
use std::collections::VecDeque;
use std::mem::swap;
use std::time::Instant;

#[inline(always)]
fn sqr_dist(mut p1: [f32; 3], p2: &[f32; 3]) -> f32 {
    p1[0] -= p2[0];
    p1[1] -= p2[1];
    p1[2] -= p2[2];
    return p1[0] * p1[0] + p1[1] * p1[1] + p1[2] * p1[2];
}

type PointIdx = ([f32; 3], usize);

struct FlatKdTree3d {
    /// Raw point data
    /// Pair representing the vertex and and the original index before being shuffled around
    data: Vec<PointIdx>,
    /// The pivots at each lair
    pivots: Vec<f32>,
}

#[inline]
fn nearest_linear(data: &Vec<[f32; 3]>, query: &[f32; 3]) -> usize {
    return data
        .iter()
        .enumerate()
        .reduce(|acc, e| {
            return if sqr_dist(*e.1, query) < sqr_dist(*acc.1, query) {
                e
            } else {
                acc
            };
        })
        .unwrap()
        .0;
}

const SAMPLE_POINTS: usize = 16;
impl FlatKdTree3d {
    fn construct_iterative(mut data: &mut Vec<([f32; 3], usize)>, pivots: &mut Vec<f32>) {
        let mut queue: VecDeque<(usize, usize, usize)> =
            VecDeque::with_capacity(data.len() + data.len() / 4);
        queue.push_back((0, data.len(), 0));
        while !queue.is_empty() {
            let (beg, end, depth) = queue.pop_front().unwrap();
            if end as i32 - beg as i32 <= 1 {
                continue;
            }
            let axis = depth % 3;
            let mid = (end - beg) / 2;
            data[beg..end].select_nth_unstable_by(mid, |a, b| a.0[axis].total_cmp(&b.0[axis]));
            let mid = mid + beg;
            pivots[mid] = data[mid].0[axis];
            queue.push_back((beg, mid, depth + 1));
            queue.push_back((mid, end, depth + 1));
        }
    }

    fn construct_section(
        data: &mut Vec<([f32; 3], usize)>,
        pivots: &mut Vec<f32>,
        beg: usize,
        end: usize,
        depth: usize,
    ) {
        if end as i32 - beg as i32 <= 1 {
            return;
        }
        let axis = depth % 3;
        data[beg..end]
            .select_nth_unstable_by((end - beg) / 2, |a, b| a.0[axis].total_cmp(&b.0[axis]));
        let mid = (end - beg) / 2 + beg;
        pivots[mid] = data[mid].0[axis];
        FlatKdTree3d::construct_section(data, pivots, beg, mid, depth + 1);
        FlatKdTree3d::construct_section(data, pivots, mid, end, depth + 1);
    }
    pub fn construct(data: &Vec<[f64; 3]>) -> Self {
        let mut pivots: Vec<f32> = vec![0.0; data.len()];
        let mut my_data: Vec<PointIdx> = data
            .iter()
            .enumerate()
            .map(|(idx, d)| ([d[0] as f32, d[1] as f32, d[2] as f32], idx))
            .collect();
        // FlatKdTree3d::construct_iterative(&mut my_data, &mut pivots);
        FlatKdTree3d::construct_section(&mut my_data, &mut pivots, 0, data.len(), 0);
        return FlatKdTree3d {
            data: my_data,
            pivots,
        };
    }

    /// Return the nearest so far
    fn nearest_inner(
        &self,
        query: &[f32; 3],
        beg: usize,
        end: usize,
        depth: usize,
    ) -> ([f32; 3], usize) {
        if end as i32 - beg as i32 <= 1 {
            return self.data[beg];
        }

        let axis = depth % 3;
        let mid = (end - beg) / 2 + beg;
        let pivot = self.pivots[mid];
        let mut nearest: PointIdx;
        if query[axis] < pivot {
            nearest = self.nearest_inner(query, beg, mid, depth + 1);
        } else {
            nearest = self.nearest_inner(query, mid, end, depth + 1);
        }

        let nearest_dist = sqr_dist(nearest.0, query);
        let mut pivot_dist = query[axis] - pivot;
        pivot_dist *= pivot_dist;

        if pivot_dist < nearest_dist {
            // only explore if the pivot is CLOSER, not if it's just as close
            // if it's just as close, then whatever point that exists could only
            // possibly be the same dist as nearest

            // we reverse the check from last time
            let pos: PointIdx;
            if query[axis] >= pivot {
                pos = self.nearest_inner(query, beg, mid, depth + 1);
            } else {
                pos = self.nearest_inner(query, mid, end, depth + 1);
            }

            let pos_dist = sqr_dist(pos.0, query);
            if pos_dist < nearest_dist {
                return pos;
            }
        }

        return nearest;
    }

    fn explore_iterative(
        &self,
        query: &[f32; 3],
        mut beg: usize,
        mut end: usize,
        mut depth: usize,
    ) -> PointIdx {
        // follow the tree down to the possible nearest
        while end as i32 - beg as i32 >= 1 {
            let mid = (end - beg) / 2 + beg;
            if self.pivots[mid] < query[depth % 3] {
                end = mid;
            } else {
                beg = mid;
            }
            depth += 1;
        }
        return self.data[beg];
    }

    fn nearest_partial_iterative(
        &self,
        query: &[f32; 3],
        beg: usize,
        end: usize,
        depth: usize,
    ) -> PointIdx {
        let axis = depth % 3;
        let mid = (end - beg) / 2 + beg;
        let pivot = self.pivots[mid];
        let nearest: PointIdx;
        if query[axis] < pivot {
            nearest = self.explore_iterative(query, beg, mid, depth + 1);
        } else {
            nearest = self.explore_iterative(query, mid, end, depth + 1);
        }

        let nearest_dist = sqr_dist(nearest.0, query);
        let mut pivot_dist = query[axis] - pivot;
        pivot_dist *= pivot_dist;

        if pivot_dist < nearest_dist {
            let pos: PointIdx;
            if query[axis] >= pivot {
                pos = self.nearest_partial_iterative(query, beg, mid, depth + 1);
            } else {
                pos = self.nearest_partial_iterative(query, mid, end, depth + 1);
            }

            let pos_dist = sqr_dist(pos.0, query);
            if pos_dist < nearest_dist {
                return pos;
            }
        }

        return nearest;
    }

    pub fn nearest(&self, query: &[f32; 3]) -> usize {
        return self.nearest_inner(query, 0, self.data.len(), 0).1;
    }
}

#[inline]
fn dot_2d(a: [f64; 2], b: [f64; 2]) -> f64 {
    return a[0] * b[0] + a[1] * b[1];
}

#[inline]
fn dot_3d_to_2d(a: [f64; 3], b: [f64; 3]) -> f64 {
    return dot_2d([a[0], a[1]], [b[0], b[1]]);
}

#[inline]
fn circumcenter(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> [f64; 2] {
    let (ax, ay) = (a[0], a[1]);
    let (bx, by) = (b[0], b[1]);
    let (cx, cy) = (c[0], c[1]);
    let ada = dot_2d(a, a);
    let bdb = dot_2d(b, b);
    let cdc = dot_2d(c, c);
    let d_inv = 1. / (2. * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)));
    return [
        d_inv * (ada * (by - cy) + bdb * (cy - ay) + cdc * (ay - by)),
        d_inv * (ada * (cx - bx) + bdb * (ax - cx) + cdc * (bx - ax)),
    ];
}

pub fn voronoi_centers_from_triangulation(
    vertices: &Vec<[f64; 2]>,
    triangulation: &Triangulation,
) -> Vec<[f64; 3]> {
    let now = Instant::now();
    // `cicumcenters[n]` corresponds to triangle n's circumcenter
    let mut circumcenters: Vec<[f64; 3]> = vec![[0., 0., f64::NAN]; triangulation.len()];
    let mut min_height = 0.0f64;

    let mut queue = VecDeque::<usize>::with_capacity(triangulation.len() / 2);
    // seed the queue
    queue.push_back(0);
    // mark the first triangle as visited. The weight will always be 0
    circumcenters[0][2] = 0.;

    // We traverse the triangulation BFS style
    // This lets the arithmetic operations become more evenly distributed
    while !queue.is_empty() {
        // grab the next node to explore
        let cur = queue.pop_front().unwrap();
        let tri_num = cur / 3;
        // iterate over it's neighbors
        for i in 0..3 {
            // for each half edge in our own triangle, calculate the neighbor's circumcenter
            // also calculate the constant for us to them
            let neighbor_he = triangulation.halfedges[cur + i];
            if neighbor_he == EMPTY {
                // no edge here
                continue;
            }
            let neighbor_tri_num = neighbor_he / 3;
            if !circumcenters[neighbor_tri_num][2].is_nan() {
                // node is visited already
                continue;
            }
            let neighbor = neighbor_tri_num * 3;
            // first calculate this node's circumcenter on the 2d plane
            // A, B, C are the tree vertices of the current triangle
            // we calculate it in such a wonky way because we want to guarantee that B is on the shared edge
            let A = vertices[triangulation.triangles[(neighbor_he - 1) % 3 + neighbor]];
            let B = vertices[triangulation.triangles[neighbor_he]];
            let C = vertices[triangulation.triangles[(neighbor_he + 1) % 3 + neighbor]];
            let cc = circumcenter(A, B, C);
            circumcenters[neighbor_tri_num][0] = cc[0];
            circumcenters[neighbor_tri_num][1] = cc[1];
            let U = circumcenters[tri_num];
            let V = circumcenters[neighbor_tri_num];
            // calculate the base edge constant, without "row echelon correction"
            // In the following formulas, U is current circumcenter and V is the neighbor circumcenter
            // B is a vertex on the shared edge between the triangles that own U and V
            // (U - V)*((U + V) - 2B) = 0
            // (U - V)*(U + V) - 2B*(U - V) = 0
            // U*U - V*V - 2B*(U - V) = 0
            // (Ux^2 + Uy^2 + Uz^2) - (Vx^2 + Vy^2 + Vz^2) - 2Bx(Ux - Vx) - 2By(Uy - Vy) - 2Bz(Uz - Vz) = 0
            // (Ux^2 + Uy^2 + Uz^2) - (Vx^2 + Vy^2 + Vz^2) - 2Bx(Ux - Vx) - 2By(Uy - Vy) = 0
            // Uz^2 - Vz^2 + [(Ux^2 + Uy^2) - (Vx^2 + Vy^2) - 2Bx(Ux - Vx) - 2By(Uy - Vy)] = 0
            // Uz^2 - Vz^2 = -[(Ux^2 + Uy^2) - (Vx^2 + Vy^2) - 2Bx(Ux - Vx) - 2By(Uy - Vy)]
            // The RHS in the last formula is the constant we need to calculate

            let mut w = -(dot_3d_to_2d(U, U)
                - dot_3d_to_2d(V, V)
                - 2. * dot_2d(B, [U[0] - V[0], U[1] - V[1]]));
            // for now we "store" the constant as the height of the circumcenter,
            // THIS IS NOT THE HEIGHT, NOR THE SQUARED HEIGHT
            // we only do this for efficiency sake
            // we also add the weight of the current, if these equations were put into matrix
            // notation this would have the effect of putting everything into reduced row echelon form
            // the first one should have a weight of 0; it's like the edge between that triangle and infinity
            w += circumcenters[tri_num][2];
            circumcenters[neighbor_tri_num][2] = w;
            // finally we update the smallest value
            min_height = min_height.min(w);
            // finally finally we add this node to the queue
            queue.push_back(neighbor);
        }
    }
    // finally construct our KD tree
    // unfortunately we have to correct the circumcenters; we couldn't do this earlier because
    // we didn't know what the smallest value was going to be
    let abs_min = min_height.abs();
    for i in 0..circumcenters.len() {
        let sqr_height = circumcenters[i][2] + abs_min;
        circumcenters[i][2] = sqr_height.sqrt();
    }

    let elapsed = now.elapsed();
    println!("Constructing tree elapsed: {:.7?}", elapsed);
    return circumcenters;
}

pub fn kd_tree_from_triangulation(
    vertices: &Vec<[f64; 2]>,
    triangulation: &Triangulation,
) -> FlatKdTree3d {
    return FlatKdTree3d::construct(&voronoi_centers_from_triangulation(vertices, triangulation));
}

#[test]
fn nn_test() {
    let mut points = vec![];
    for x in 0..100 {
        for y in 0..1000 {
            points.push([
                thread_rng().gen_range(-X_VIEW_DIST_M..X_VIEW_DIST_M),
                thread_rng().gen_range(-Z_VIEW_DIST_M..Z_VIEW_DIST_M),
            ]);
        }
    }
    let delaunay_points: Vec<Point> = points
        .iter()
        .map(|v| Point {
            x: (*v)[0],
            y: (*v)[1],
        })
        .collect();
    let triangulation = triangulate(&delaunay_points);
    let tree = kd_tree_from_triangulation(&points, &triangulation);
    for pi in 0..100 {
        let query = &[
            thread_rng().gen_range(-X_VIEW_DIST_M..X_VIEW_DIST_M) as f32,
            thread_rng().gen_range(-Z_VIEW_DIST_M..Z_VIEW_DIST_M) as f32,
            0.,
        ];
        let nearest = tree.nearest(query);
        // println!("ORIG IDX {}", nearest);
        let mut smallest_dist: f32 = sqr_dist(tree.data[0].0, query);
        let mut orig_idx = 0;
        let mut cur_idx = 0;
        // for (i, p) in tree.data.iter().enumerate() {
        //     println!("CUR IDX {i} > ORIG IDX {} | {} {} {} | {} ", p.1, p.0[0], p.0[1], p.0[2], sqr_dist(p.0, query));
        // }
        //
        // fn print_pivot(p: &Vec<f32>, b: usize, e: usize, d: usize) {
        //     let mut leading = "----------------------------".to_string();
        //     for i in 0..d {
        //         leading.remove(0);
        //         leading.remove(0);
        //         leading.remove(0);
        //         leading.remove(0);
        //     }
        //     leading += ">";
        //     if e as i32- b as i32 <= 1 {
        //         println!("{} LEAF {}", leading, b)
        //     } else {
        //         let m = (e - b)/2 + b;
        //         print_pivot(p, b, m, d + 1);
        //         println!("{} {} | CUR {} of [{}, {}) | {} | {}", leading, p[m], m, b, e, d, d % 3);
        //         print_pivot(p, m, e, d + 1);
        //     }
        // }
        //
        // print_pivot(&tree.pivots, 0, tree.pivots.len(), 0);
        for (i, p) in tree.data.iter().enumerate() {
            let d = sqr_dist(p.0, query);
            // println!("{}", d);
            if d < smallest_dist {
                smallest_dist = d;
                orig_idx = p.1;
                cur_idx = i;
            }
        }
        assert_eq!(orig_idx, nearest);
        // println!("Point {} matches", pi);
        // println!("CUR IDX {} > ORIG IDX {}", cur_idx, orig_idx);
    }
}

#[test]
fn create_and_find_bench() {
    let mut queries = vec![];
    let mut points = vec![];
    for x in 0..100 {
        for y in 0..1000 {
            points.push([
                thread_rng().gen_range(-X_VIEW_DIST_M..X_VIEW_DIST_M),
                thread_rng().gen_range(-Z_VIEW_DIST_M..Z_VIEW_DIST_M),
            ]);
            if queries.len() < 10000 {
                queries.push([
                    thread_rng().gen_range(-X_VIEW_DIST_M..X_VIEW_DIST_M) as f32,
                    thread_rng().gen_range(-Z_VIEW_DIST_M..Z_VIEW_DIST_M) as f32,
                    0.,
                ]);
            }
        }
    }
    let total = Instant::now();
    let now = Instant::now();
    let delaunay_points: Vec<Point> = points
        .iter()
        .map(|v| Point {
            x: (*v)[0],
            y: (*v)[1],
        })
        .collect();
    let triangulation = triangulate(&delaunay_points);
    let elapsed = now.elapsed();
    println!("Constructing triangulation elapsed: {:.7?}", elapsed);

    let now = Instant::now();
    let tree = kd_tree_from_triangulation(&points, &triangulation);
    let elapsed = now.elapsed();
    println!("Constructing kd and circumcenters elapsed: {:.7?}", elapsed);

    let now = Instant::now();
    for query in &queries {
        let nearest = tree.nearest(query);
    }
    let elapsed = now.elapsed();
    let total = total.elapsed();
    println!("Finding 100k nearest elapsed: {:.2?}", elapsed);
    println!("Total elapsed: {:.2?}", total);

    let total = Instant::now();
    let now = Instant::now();
    let delaunay_points: Vec<Point> = points
        .iter()
        .map(|v| Point {
            x: (*v)[0],
            y: (*v)[1],
        })
        .collect();
    let triangulation = triangulate(&delaunay_points);
    let elapsed = now.elapsed();
    println!("Constructing triangulation elapsed: {:.7?}", elapsed);

    let now = Instant::now();
    let centers = voronoi_centers_from_triangulation(&points, &triangulation)
        .iter()
        .map(|e| [e[0] as f32, e[1] as f32, e[2] as f32])
        .collect();
    let elapsed = now.elapsed();
    println!("Constructing circumcenters elapsed: {:.7?}", elapsed);

    let now = Instant::now();
    for query in &queries {
        let nearest = nearest_linear(&centers, query);
    }
    let elapsed = now.elapsed();
    let total = total.elapsed();
    println!("Finding 100k nearest elapsed: {:.2?}", elapsed);
    println!("Total elapsed: {:.2?}", total);
}
