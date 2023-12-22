use bitvec::array::BitArray;
use bitvec::{bitarr, bits, bitvec};
use delaunator::{triangulate, Point, Triangulation};
use std::mem::swap;
use std::rc::Rc;
use std::time::Instant;
use rand::{Rng, thread_rng};
use crate::terrain_render::{X_VIEW_DIST_M, Z_VIEW_DIST_M};

fn triangle_intersects_axis(
    vertices: &Vec<[f64; 2]>,
    triangle: &[usize; 3],
    pivot: f64,
    axis: usize,
) -> i8 {
    let mut neg = false;
    let mut pos = false;
    for v in triangle {
        if vertices[*v][axis] < pivot {
            neg = true;
        } else {
            pos = true;
        }
    }
    return if neg == pos {
        -1
    } else if neg {
        0
    } else {
        1
    };
}

#[inline]
/// Code copied and adapted from ChatGPT 3.5
/// https://chat.openai.com/share/aea691d2-36fd-44f0-a600-3c3f3533f216
/// This returns where each section ends
fn sort_triangles_around_axis(
    vertices: &Vec<[f64; 2]>,
    triangles: &mut [[usize; 3]],
    pivot: f64,
    axis: usize,
) -> (usize, usize) {
    let mut type_1_end = 0;
    let mut type_2_end = 0;
    let mut type_3_end: i32 = triangles.len() as i32 - 1;

    while type_2_end as i32 <= type_3_end {
        let cmp = triangle_intersects_axis(vertices, &triangles[type_2_end], pivot, axis);
        if cmp == -1 {
            triangles.swap(type_1_end, type_2_end);
            type_1_end += 1;
            type_2_end += 1;
        } else if cmp == 0 {
            type_2_end += 1
        } else {
            triangles.swap(type_2_end, type_3_end as usize);
            type_3_end -= 1;
        }
    }
    return (type_1_end, type_2_end);
}

struct PregrackeHeirarchy {
    /// List of triangles, in order
    data: Vec<usize>,
    // /// Metadata about each of the triangle sets
    // meta: Vec<usize>,
}

impl PregrackeHeirarchy {
    fn construct_inner(
        beg: usize,
        end: usize,
        vertices: &Vec<[f64; 2]>,
        verts: &mut Vec<(usize, [f64; 2])>,
        mut triangles: &mut Vec<[usize; 3]>,
        depth: usize,
    ) {
        println!("{} {} {}", beg, end, depth);
        if end as i32 - beg as i32 <= 1 {
            return;
        }
        let axis = depth % 2;
        let mid = (end - beg) / 2;
        verts[beg..end].select_nth_unstable_by(mid, |a, b| a.1[axis].total_cmp(&b.1[axis]));
        let pivot = verts[mid + beg];
        // now sort the triangles s.t. intersects is lowest, left is middle, and right is highest
        let (shared_end, left_end) =
            sort_triangles_around_axis(vertices, &mut triangles[beg..end], pivot.1[axis], axis);
        let shared_beg = beg;
        let right_end = end;
        // finally recurse on our three groups
        // make sure we aren't repeating our selves
        // if the shared is exactly equal to the original, don't recurse
        if shared_beg == beg && shared_end == end {
            return;
        }
        PregrackeHeirarchy::construct_inner(
            shared_beg,
            shared_end,
            vertices,
            verts,
            triangles,
            depth + 1,
        );
        PregrackeHeirarchy::construct_inner(
            shared_end,
            left_end,
            vertices,
            verts,
            triangles,
            depth + 1,
        );
        PregrackeHeirarchy::construct_inner(
            left_end,
            right_end,
            vertices,
            verts,
            triangles,
            depth + 1,
        );
    }

    fn construct(vertices: &Vec<[f64; 2]>, triangulation: &Triangulation) -> Self {
        // first find a suitable midline
        let mut verts: Vec<(usize, [f64; 2])> = vertices
            .iter()
            .enumerate()
            .map(|(idx, p)| (idx, *p))
            .collect();
        let mut triangles: Vec<[usize; 3]> = vec![[0, 0, 0]; triangulation.len()];
        for i in 0..triangulation.len() {
            triangles[i] = [
                triangulation.triangles[3 * i],
                triangulation.triangles[3 * i + 1],
                triangulation.triangles[3 * i + 2],
            ];
        }
        let depth = 0;
        let beg = 0;
        let end = verts.len();
        PregrackeHeirarchy::construct_inner(
            beg,
            end,
            vertices,
            &mut verts,
            &mut triangles,
            depth,
        );
        return PregrackeHeirarchy {
            data: triangles.into_iter().flatten().collect()
        };
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
    let delaunay_points: Vec<Point> = points.iter().map(|v| Point {
        x: (*v)[0],
        y: (*v)[1],
    }).collect();
    let triangulation = triangulate(&delaunay_points);
    let elapsed = now.elapsed();
    println!("Constructing triangulation elapsed: {:.7?}", elapsed);

    let now = Instant::now();
    let ph = PregrackeHeirarchy::construct(&points, &triangulation);
    let elapsed = now.elapsed();
    println!("Constructing kd and circumcenters elapsed: {:.7?}", elapsed);
}