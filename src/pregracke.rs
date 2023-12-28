//use crate::terrain_render::{X_VIEW_DIST_M, Z_VIEW_DIST_M};
use bitvec::array::BitArray;
use bitvec::{bitarr, bits, bitvec};
use delaunator::{triangulate, Point, Triangulation};
use image::{Pixel, Rgba, RgbaImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_line_segment_mut, draw_text_mut};
use imageproc::drawing::{Blend, Canvas};
use rand::{thread_rng, Rng};
use server::util::{intersects_tri, lin_map};
use std::mem::swap;
use std::rc::Rc;
use std::time::Instant;
use rusttype::{Font, Scale};

const X_VIEW_DIST_M: f64 = 10.;
const Z_VIEW_DIST_M: f64 = 10.;

fn spaces(num: usize) -> String {
    let mut s = "".to_string();
    for i in 0..num {
        s += "  ";
    }
    return s;
}

fn triangle_intersects_axis(triangle: &[[f64; 2]; 3], pivot: f64, axis: usize) -> i8 {
    let mut neg = false;
    let mut pos = false;
    for p in triangle {
        if p[axis] < pivot {
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

fn bounding_box(tri: &[[f64; 2]; 3]) -> [[f64; 2]; 2] {
    let v1 = tri[0];
    let v2 = tri[1];
    let v3 = tri[2];
    return [
        [v1[0].min(v2[0]).min(v3[0]), v1[1].min(v2[1]).min(v3[1])],
        [v1[0].max(v2[0]).max(v3[0]), v1[1].max(v2[1]).max(v3[1])],
    ];
}

fn centroid(poly: &[[f64; 2]]) -> [f64; 2] {
    let mut avg = [0., 0.];
    for e in poly {
        avg[0] += e[0];
        avg[1] += e[1];
    }
    let l = poly.len() as f64;
    return [avg[0] / l, avg[1] / l];
}

struct PregrackeHeirarchy {
    /// Vertices of the triangulation
    vertices: Vec<[f64; 2]>,
    /// List of triangles, in order
    data: Vec<[usize; 3]>,
    /// Metadata about each of the triangle sets
    meta: Vec<(usize, f64)>,
}

#[inline]
fn get_tri(triangles: &Vec<[usize; 3]>, vertices: &Vec<[f64; 2]>, idx: usize) -> [[f64; 2]; 3] {
    return [
        vertices[triangles[idx][0]],
        vertices[triangles[idx][1]],
        vertices[triangles[idx][2]],
    ];
}

#[inline]
/// Code copied and adapted from ChatGPT 3.5
/// https://chat.openai.com/share/aea691d2-36fd-44f0-a600-3c3f3533f216
/// This returns where each section ends
fn sort_triangles_around_axis(
    beg: usize,
    end: usize,
    mut triangles: &mut Vec<[usize; 3]>,
    vertices: &Vec<[f64; 2]>,
    pivot: f64,
    axis: usize,
) -> (usize, usize) {
    let mut type_1_end = beg;
    let mut type_2_end = beg;
    let mut type_3_end: i32 = end as i32 - 1;

    while type_2_end as i32 <= type_3_end {
        let cmp = triangle_intersects_axis(&get_tri(&triangles, vertices, type_2_end), pivot, axis);
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

impl PregrackeHeirarchy {
    #[inline]
    fn get_tri(&self, idx: usize) -> [[f64; 2]; 3] {
        return get_tri(&self.data, &self.vertices, idx);
    }

    fn construct_inner(
        beg: usize,
        end: usize,
        vertices: &Vec<[f64; 2]>,
        mut centroids: &mut Vec<[f64; 2]>,
        mut triangles: &mut Vec<[usize; 3]>,
        mut meta: &mut Vec<(usize, f64)>,
        depth: usize,
    ) {
        //print!("{} {} {}\n", beg, end, depth);
        if end as i32 - beg as i32 <= 1 {
            return;
        }

        let axis = depth % 2;
        let mid = (end - beg) / 2;
        // we select a pivot by selecting the median of all the vertices of the bounding boxes
        centroids[beg..end].select_nth_unstable_by(mid, |a, b| a[axis].total_cmp(&b[axis]));
        let pivot = centroids[mid];
        // now sort the triangles s.t. intersects is lowest, left is middle, and right is highest
        let (shared_end, left_end) =
            sort_triangles_around_axis(beg, end, &mut triangles, &vertices, pivot[axis], axis);
        let shared_beg = beg;
        let right_end = end;
        //println!("{}{} | {} {} | {} {} | {} {} | {}", spaces(depth), beg, shared_beg, shared_end, shared_end, left_end, left_end, right_end, end);
        // finally recurse on our three groups
        // make sure we aren't repeating our selves
        // if the shared is exactly equal to the original, don't recurse
        if shared_beg == beg && shared_end == end {
            return;
        }

        println!(
            "{}Meta {} {} {}, {} {}",
            spaces(depth),
            depth,
            beg,
            end,
            depth + beg,
            end - beg
        );
        meta[beg + depth] = (end - beg, pivot[axis]);

        if shared_beg != beg || shared_end != end {
            //print!("{}Shared ", spaces(depth+1));
            //println!("{}Shared {} {} {}", spaces(depth+1), shared_beg, shared_end, depth + 1);
            PregrackeHeirarchy::construct_inner(
                shared_beg,
                shared_end,
                &vertices,
                centroids,
                triangles,
                meta,
                depth + 1,
            );
            //meta[shared_beg + depth] = shared_end - shared_beg;
        }
        if shared_end != beg || left_end != end {
            //print!("{}Left ", spaces(depth+1));
            //println!("{}Left {} {} {}", spaces(depth+1), shared_end, left_end, depth + 1);
            PregrackeHeirarchy::construct_inner(
                shared_end,
                left_end,
                &vertices,
                centroids,
                triangles,
                meta,
                depth + 1,
            );
            //meta[shared_end + depth] = left_end - shared_end;
        }
        if left_end != beg || right_end != end {
            //print!("{}Right ", spaces(depth+1));
            //println!("{}Right {} {} {}", spaces(depth+1), left_end, right_end, depth + 1);
            PregrackeHeirarchy::construct_inner(
                left_end,
                right_end,
                &vertices,
                centroids,
                triangles,
                meta,
                depth + 1,
            );
            //meta[left_end + depth] = right_end - left_end;
        }
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
        let mut meta: Vec<(usize, f64)> =
            vec![(0, 0.); triangulation.len() + triangulation.len() / 2];
        let mut centroids: Vec<[f64; 2]> = triangles
            .iter()
            .map(|t| {
                let v1 = vertices[t[0]];
                let v2 = vertices[t[1]];
                let v3 = vertices[t[2]];
                return centroid(&[v1, v2, v3]);
            })
            .collect();
        let depth = 0;
        let beg = 0;
        let end = triangulation.len();
        PregrackeHeirarchy::construct_inner(
            beg,
            end,
            &vertices,
            &mut centroids,
            &mut triangles,
            &mut meta,
            depth,
        );
        return PregrackeHeirarchy {
            vertices: vertices.clone(),
            data: triangles.clone(),
            meta,
        };
    }

    fn locate_inner(&self, query: &[f64; 2], beg: usize, end: usize, depth: usize) -> usize {
        // make sure there's actually things to check
        let len = end as i32 - beg as i32;
        println!(
            "{}{} a:{} beg: {} end: {} len: {}",
            spaces(depth),
            depth,
            depth % 2,
            beg,
            end,
            len
        );
        if len <= 0 {
            return usize::MAX;
        } else if len == 1 {
            if intersects_tri(query, &self.get_tri(beg)) {
                return beg;
            } else {
                return usize::MAX;
            }
        }
        let axis = depth % 2;
        let total = self.meta[beg + depth].0;
        // if this is a leaf node (or effective leaf node)
        // we can search the children right now
        // we won't need to check any of the children if they
        if total == self.meta[beg + depth + 1].0 || total == 0 {
            //if bounding_box(&self.get_tri(beg))[0][axis] <= query[axis] && query[axis] <= bounding_box(&self.get_tri(end-1))[1][axis] {
            for i in beg..end {
                if intersects_tri(&query, &self.get_tri(i)) {
                    return i;
                }
            }
            //} else {
            return usize::MAX;
            //}
        }

        // if this thing is the leaf, then don't do anything.
        let shared_len = self.meta[beg + depth + 1].0;
        let left_len = self.meta[beg + depth + shared_len + 1].0;
        println!(
            "{}{} a:{} total: {} beg: {} end: {}  shared: {} left: {}",
            spaces(depth),
            depth,
            axis,
            total,
            beg,
            end,
            shared_len,
            left_len
        );
        let right_len = (end - beg) - (shared_len + left_len);
        let shared_beg = beg;
        let shared_end = shared_beg + shared_len;
        let left_beg = shared_end;
        let left_end = left_beg + left_len;
        let right_beg = left_end;
        let right_end = right_beg + right_len;
        println!("{}{} a:{} total: {} sl: {} ll: {} rl: {} | beg: {} sb: {} se: {} lb: {} le: {} rb: {} re: {} end: {}", spaces(depth), depth, axis, total, 
                 shared_len, left_len, right_len,
                 beg, shared_beg, shared_end, left_beg, left_end, right_beg, right_end, end);

        println!(
            "{}{} SHARED a:{} total: {} beg: {} sbeg: {} send: {} end: {}",
            spaces(depth),
            depth,
            axis,
            total,
            beg,
            shared_beg,
            shared_end,
            end
        );
        let shared_res = self.locate_inner(query, shared_beg, shared_end, depth + 1);
        if shared_res != usize::MAX {
            return shared_res;
        }

        if query[axis] < self.meta[beg + depth].1 {
            // check the left branch
            println!(
                "{}{} LEFT a:{} total: {} beg: {} lbeg: {} lend: {} end: {}",
                spaces(depth),
                depth,
                axis,
                total,
                beg,
                left_beg,
                left_end,
                end
            );
            return self.locate_inner(query, left_beg, left_end, depth + 1);
        } else {
            println!(
                "{}{} RIGHT a:{} total: {} beg: {} rbeg: {} rend: {} end: {}",
                spaces(depth),
                depth,
                axis,
                total,
                beg,
                right_beg,
                right_end,
                end
            );
            return self.locate_inner(query, right_beg, right_end, depth + 1);
        }
    }

    fn locate(&self, query: &[f64; 2]) -> usize {
        return self.locate_inner(query, 0, self.data.len(), 0);
    }
}

#[test]
fn create_and_find_bench() {
    let mut queries = vec![];
    let mut points = vec![];
    for x in 0..4 {
        for y in 0..4 {
            points.push([
                thread_rng().gen_range(-X_VIEW_DIST_M..X_VIEW_DIST_M),
                thread_rng().gen_range(-Z_VIEW_DIST_M..Z_VIEW_DIST_M),
            ]);
            if queries.len() < 10000 {
                queries.push([
                    thread_rng().gen_range(-X_VIEW_DIST_M..X_VIEW_DIST_M),
                    thread_rng().gen_range(-Z_VIEW_DIST_M..Z_VIEW_DIST_M),
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
    let ph = PregrackeHeirarchy::construct(&points, &triangulation);
    let elapsed = now.elapsed();
    println!("Constructing heirarchy elapsed: {:.7?}", elapsed);
    let mut triangles: Vec<[usize; 3]> = vec![[0, 0, 0]; triangulation.len()];
    for i in 0..triangulation.len() {
        triangles[i] = [
            triangulation.triangles[3 * i],
            triangulation.triangles[3 * i + 1],
            triangulation.triangles[3 * i + 2],
        ];
    }
    draw_debug("original_tris.png", &points, &triangles);
    draw_debug("modified_tris.png", &ph.vertices, &ph.data);
    println!("Splits:");
    for i in ph.meta.iter().enumerate() {
        println!("{:?}", i);
    }
    println!("Vertices:");
    for i in ph.vertices.iter().enumerate() {
        println!("{:?}", i);
    }
    println!("Triangles:");
    for i in ph.data.iter().enumerate() {
        println!("{:?}", i);
    }
    for q in &queries {
        let mut found = usize::MAX;
        for (i, tri) in ph.data.iter().enumerate() {
            // first locate the point
            if intersects_tri(q, &ph.get_tri(i)) {
                found = i;
                break;
            }
        }
        assert_eq!(found, ph.locate(q));
    }
}

fn draw_debug(path: &str, vertices: &Vec<[f64; 2]>, triangles: &Vec<[usize; 3]>) {
    // Background color
    let white = Rgba([255u8, 255u8, 255u8, 255u8]);

    // Drawing color
    let red = Rgba([255u8, 0u8, 0u8, 255u8]);
    let light_red = Rgba([255u8, 150u8, 150u8, 255u8]);
    let blue = Rgba([0u8, 0u8, 255u8, 255u8]);
    let light_blue = Rgba([150u8, 150u8, 255u8, 255u8]);

    let dim = 1024;

    // The implementation of Canvas for GenericImage overwrites existing pixels
    let mut image = RgbaImage::from_pixel(dim, dim, white);

    let font = Vec::from(include_bytes!("/Library/Fonts/Arial Unicode.ttf") as &[u8]);
    let font = Font::try_from_vec(font).unwrap();

    let height = 24.;

    let scale = Scale {
        x: height,
        y: height,
    };

    for [aidx, bidx, cidx] in triangles {
        let a = vertices[*aidx];
        let b = vertices[*bidx];
        let c = vertices[*cidx];
        let a = (
            lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, a[0]) as f32,
            lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, a[1]) as f32,
        );
        let b = (
            lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, b[0]) as f32,
            lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, b[1]) as f32,
        );
        let c = (
            lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, c[0]) as f32,
            lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, c[1]) as f32,
        );
        let ai = (a.0.round() as i32, a.1.round() as i32);
        let bi = (b.0.round() as i32, b.1.round() as i32);
        let ci = (c.0.round() as i32, c.1.round() as i32);
        let at = format!("{}", *aidx);
        let bt = format!("{}", *bidx);
        let ct = format!("{}", *cidx);
        draw_line_segment_mut(&mut image, a.clone(), b.clone(), red);
        draw_line_segment_mut(&mut image, b.clone(), c.clone(), red);
        draw_line_segment_mut(&mut image, c.clone(), a.clone(), red);
    }

    for [aidx, bidx, cidx] in triangles {
        let a = vertices[*aidx];
        let b = vertices[*bidx];
        let c = vertices[*cidx];
        let a = (
            lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, a[0]) as f32,
            lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, a[1]) as f32,
        );
        let b = (
            lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, b[0]) as f32,
            lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, b[1]) as f32,
        );
        let c = (
            lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, c[0]) as f32,
            lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, c[1]) as f32,
        );
        let ai = (a.0.round() as i32, a.1.round() as i32);
        let bi = (b.0.round() as i32, b.1.round() as i32);
        let ci = (c.0.round() as i32, c.1.round() as i32);
        let at = format!("{}", *aidx);
        let bt = format!("{}", *bidx);
        let ct = format!("{}", *cidx);
        draw_filled_circle_mut(&mut image, ai, 5, light_blue);
        draw_filled_circle_mut(&mut image, bi, 5, light_blue);
        draw_filled_circle_mut(&mut image, ci, 5, light_blue);
    }

    for [aidx, bidx, cidx] in triangles {
        let a = vertices[*aidx];
        let b = vertices[*bidx];
        let c = vertices[*cidx];
        let a = (
            lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, a[0]) as f32,
            lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, a[1]) as f32,
        );
        let b = (
            lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, b[0]) as f32,
            lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, b[1]) as f32,
        );
        let c = (
            lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, c[0]) as f32,
            lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, c[1]) as f32,
        );
        let ai = (a.0.round() as i32, a.1.round() as i32);
        let bi = (b.0.round() as i32, b.1.round() as i32);
        let ci = (c.0.round() as i32, c.1.round() as i32);
        let at = format!("{}", *aidx);
        let bt = format!("{}", *bidx);
        let ct = format!("{}", *cidx);
        draw_text_mut(&mut image, blue, ai.0, ai.1, scale, &font, &at.as_str()); 
        draw_text_mut(&mut image, blue, bi.0, bi.1, scale, &font, &bt.as_str()); 
        draw_text_mut(&mut image, blue, ci.0, ci.1, scale, &font, &ct.as_str()); 
    }

    for (i, [aidx, bidx, cidx]) in triangles.iter().enumerate() {
        let a = vertices[*aidx];
        let b = vertices[*bidx];
        let c = vertices[*cidx];
        let centroid = [(a[0] + b[0] + c[0]) / 3., (a[1] + b[1] + c[1]) / 3.]; 
        let cen = (
            lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, centroid[0]) as f32,
            lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, centroid[1]) as f32,
        );
        let ci = (cen.0.round() as i32, cen.1.round() as i32);
        let ct = format!("{}", i);
        draw_text_mut(&mut image, red, ci.0, ci.1, scale, &font, &ct.as_str()); 
    }

    for (i, [aidx, bidx, cidx]) in triangles.iter().enumerate() {
        let a = vertices[*aidx];
        let b = vertices[*bidx];
        let c = vertices[*cidx];
        let centroid = [(a[0] + b[0] + c[0]) / 3., (a[1] + b[1] + c[1]) / 3.]; 
        let cen = (
            lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, centroid[0]) as f32,
            lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, centroid[1]) as f32,
        );
        let ci = (cen.0.round() as i32, cen.1.round() as i32);
        draw_filled_circle_mut(&mut image, ci, 5, light_red);
    }

    image.save(path).unwrap();
}
