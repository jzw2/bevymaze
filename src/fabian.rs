use crate::terrain_render::{X_VIEW_DIST_M, Z_VIEW_DIST_M};
use delaunator::{next_halfedge, prev_halfedge, triangulate, Point, Triangulation, EMPTY};
use image::{Rgba, RgbaImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_line_segment_mut, draw_text_mut};
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use rusttype::{Font, Scale};
use server::util::{lin_map, lin_map32, orientation, point_in_triangle, point_in_triangle_bary};
use std::ops::Range;
use std::time::Instant;

#[inline]
fn get_tri(triangles: &Vec<[usize; 3]>, vertices: &Vec<[f64; 2]>, idx: usize) -> [[f64; 2]; 3] {
    return [
        vertices[triangles[idx][0]],
        vertices[triangles[idx][1]],
        vertices[triangles[idx][2]],
    ];
}

struct EdgeGuesser {
    domain: (Range<f32>, Range<f32>),
    guesses: Vec<[usize; 256]>,
}

impl EdgeGuesser {
    /// We walk over the grid and continuously guess the best triangle
    fn build(
        domain: (Range<f32>, Range<f32>),
        vertices: &Vec<[f32; 2]>,
        triangulation: &Triangulation,
    ) -> Self {
        let mut guesses = vec![[0usize; 256]; 256];
        let mut last_guess = 0;

        for (x, guess_row) in guesses.iter_mut().enumerate() {
            for (y, guess) in guess_row.iter_mut().enumerate() {
                let xp = lin_map32(0., 256., domain.0.start, domain.0.end, x as f32);
                let yp = lin_map32(0., 256., domain.1.start, domain.1.end, y as f32);
                *guess = find_triangle(vertices, triangulation, &[xp, yp], last_guess, usize::MAX);
                last_guess = *guess * 3;
            }
            // set the last guess to the first element of this row, a.k.a. the element above the first element in the next row
            last_guess = guess_row[0];
        }

        return EdgeGuesser { domain, guesses };
    }

    fn guess(&self, p: &[f32; 2]) -> usize {
        let [x, y] = p;
        let xi = lin_map32(self.domain.0.start, self.domain.0.end, 0., 256., *x).round() as usize;
        let yi = lin_map32(self.domain.1.start, self.domain.1.end, 0., 256., *y).round() as usize;
        let xi = xi.min(255).max(0);
        let yi = yi.min(255).max(0);
        return self.guesses[xi][yi] * 3;
    }
}

/// Copied and adapted from
/// https://observablehq.com/@mootari/delaunay-findtriangle
pub fn find_triangle(
    vertices: &Vec<[f32; 2]>,
    triangulation: &Triangulation,
    p: &[f32; 2],
    edge: usize,
    mut limit: usize,
) -> usize {
    if edge == usize::MAX {
        return edge;
    }
    // coords is required for delaunator compatibility.
    let triangles = &triangulation.triangles;
    let halfedges = &triangulation.halfedges;
    let mut current = edge;
    let mut start = current;
    let mut n = 0usize;

    loop {
        if limit <= 0 {
            return current;
        };
        limit -= 1;

        let next = next_halfedge(current);
        let pc = triangles[current];
        let pn = triangles[next];

        if orientation(&vertices[pc], &vertices[pn], &p) >= 0. {
            current = next;
            if start == current {
                return current;
            }
        } else {
            if halfedges[current] == usize::MAX {
                return current;
            }
            current = halfedges[current];
            n += 1;
            current = if n % 2 != 0 {
                next_halfedge(current)
            } else {
                prev_halfedge(current)
            };
            start = current;
        }
    }
}

// const X_VIEW_DIST_M: f64 = 10.;
// const Z_VIEW_DIST_M: f64 = 10.;

// #[test]
// fn find_bench() {
//     let mut queries = vec![];
//     let mut points = vec![];
//     let mut r = StdRng::seed_from_u64(222);
//     fn transform(p: &mut [f64; 2]) -> &[f64; 2] {
//         let mut mag = (p[0] * p[0] + p[1] * p[1]).sqrt();
//         p[0] /= mag;
//         p[1] /= mag;
//         mag = mag.sinh();
//         p[0] *= mag;
//         p[1] *= mag;
//         return p;
//     }
//     let x_range = -X_VIEW_DIST_M.asinh()..X_VIEW_DIST_M.asinh();
//     let z_range = -Z_VIEW_DIST_M.asinh()..Z_VIEW_DIST_M.asinh();
//     for x in 0..100 {
//         for y in 0..1000 {
//             points.push(*transform(&mut [
//                 r.gen_range(x_range.clone()),
//                 r.gen_range(z_range.clone()),
//             ]));
//             if queries.len() < 50000 {
//                 queries.push(*transform(&mut [
//                     r.gen_range(x_range.clone()),
//                     r.gen_range(z_range.clone()),
//                 ]));
//             }
//         }
//     }
//     let total = Instant::now();
//     let now = Instant::now();
//     let delaunay_points: Vec<Point> = points
//         .iter()
//         .map(|v| Point {
//             x: (*v)[0],
//             y: (*v)[1],
//         })
//         .collect();
//     let triangulation = triangulate(&delaunay_points);
//     let elapsed = now.elapsed();
//     println!("Constructing triangulation elapsed: {:.7?}", elapsed);
//
//     let mut triangles: Vec<[usize; 3]> = vec![[0, 0, 0]; triangulation.len()];
//     for i in 0..triangulation.len() {
//         triangles[i] = [
//             triangulation.triangles[3 * i],
//             triangulation.triangles[3 * i + 1],
//             triangulation.triangles[3 * i + 2],
//         ];
//     }
//     let now = Instant::now();
//     let x_range = -X_VIEW_DIST_M..X_VIEW_DIST_M;
//     let z_range = -Z_VIEW_DIST_M..Z_VIEW_DIST_M;
//     let guesser = EdgeGuesser::build((x_range, z_range), &points, &triangulation);
//     let elapsed = now.elapsed();
//     println!("Constructing guesser elapsed: {:.7?}", elapsed);
//
//     let now = Instant::now();
//     // let mut e = vec![];
//     for q in &queries {
//         find_triangle(&points,  &triangulation, &q, guesser.guess(&q),  50);
//     }
//     let elapsed = now.elapsed();
//     println!("Locating elapsed: {:.7?}", elapsed);
// }
//
// #[test]
// fn find_test() {
//     let mut queries = vec![];
//     let mut points = vec![];
//     let mut r = StdRng::seed_from_u64(222);
//     fn transform(p: &mut [f64; 2]) -> &[f64; 2] {
//         let mut mag = (p[0] * p[0] + p[1] * p[1]).sqrt();
//         p[0] /= mag;
//         p[1] /= mag;
//         mag = mag.sinh();
//         p[0] *= mag;
//         p[1] *= mag;
//         return p;
//     }
//     for x in 0..100 {
//         for y in 0..100 {
//             let x_range = -X_VIEW_DIST_M.asinh()..X_VIEW_DIST_M.asinh();
//             let z_range = -Z_VIEW_DIST_M.asinh()..Z_VIEW_DIST_M.asinh();
//             points.push(*transform(&mut [
//                 r.gen_range(x_range.clone()),
//                 r.gen_range(z_range.clone()),
//             ]));
//             if queries.len() < 100 {
//                 queries.push(*transform(&mut [
//                     r.gen_range(x_range.clone()),
//                     r.gen_range(z_range.clone()),
//                 ]));
//             }
//         }
//     }
//     let total = Instant::now();
//     let now = Instant::now();
//     let delaunay_points: Vec<Point> = points
//         .iter()
//         .map(|v| Point {
//             x: (*v)[0],
//             y: (*v)[1],
//         })
//         .collect();
//     let triangulation = triangulate(&delaunay_points);
//     let elapsed = now.elapsed();
//     println!("Constructing triangulation elapsed: {:.7?}", elapsed);
//
//     let mut triangles: Vec<[usize; 3]> = vec![[0, 0, 0]; triangulation.len()];
//     for i in 0..triangulation.len() {
//         triangles[i] = [
//             triangulation.triangles[3 * i],
//             triangulation.triangles[3 * i + 1],
//             triangulation.triangles[3 * i + 2],
//         ];
//     }
//
//     // let now = Instant::now();
//     // for q in &queries {
//     //     find_triangle(&points, &triangulation, &q,0, i64::MAX);
//     // }
//     // let elapsed = now.elapsed();
//     // println!("Locating elapsed: {:.7?}", elapsed);
//
//     for (i, q) in queries.iter().enumerate() {
//         let mut found = usize::MAX;
//         for (i, tri) in triangles.iter().enumerate() {
//             // first locate the point
//             if point_in_triangle(q, &get_tri(&triangles, &points, i)) {
//                 found = i;
//                 break;
//             }
//         }
//         // let mut exp = vec![];
//         let res = find_triangle(&points, &triangulation, q, 0, /*&mut exp,*/ i64::MAX);
//         // draw_debug(
//         //     format!("fabian{}.png", i).as_str(),
//         //     Some((&exp, &q)),
//         //     &triangulation,
//         //     &points,
//         //     &triangles,
//         // );
//         assert_eq!(found, res);
//     }
// }
//
// fn draw_debug(
//     path: &str,
//     q: Option<(&Vec<usize>, &[f64; 2])>,
//     triangulation: &Triangulation,
//     vertices: &Vec<[f64; 2]>,
//     triangles: &Vec<[usize; 3]>,
// ) {
//     // Background color
//     let white = Rgba([255u8, 255u8, 255u8, 255u8]);
//
//     // Drawing color
//     let red = Rgba([255u8, 0u8, 0u8, 255u8]);
//     let light_red = Rgba([255u8, 150u8, 150u8, 255u8]);
//     let blue = Rgba([0u8, 0u8, 255u8, 255u8]);
//     let light_blue = Rgba([150u8, 150u8, 255u8, 255u8]);
//     let green = Rgba([0u8, 255u8, 0u8, 255u8]);
//     let purple = Rgba([255u8, 0u8, 255u8, 255u8]);
//
//     let dim = 1024;
//
//     // The implementation of Canvas for GenericImage overwrites existing pixels
//     let mut image = RgbaImage::from_pixel(dim, dim, white);
//
//     let font =
//         Vec::from(include_bytes!("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") as &[u8]);
//     let font = Font::try_from_vec(font).unwrap();
//
//     let height = 24.;
//
//     let scale = Scale {
//         x: height,
//         y: height,
//     };
//
//     for [aidx, bidx, cidx] in triangles {
//         let a = vertices[*aidx];
//         let b = vertices[*bidx];
//         let c = vertices[*cidx];
//         let a = (
//             lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, a[0]) as f32,
//             lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, a[1]) as f32,
//         );
//         let b = (
//             lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, b[0]) as f32,
//             lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, b[1]) as f32,
//         );
//         let c = (
//             lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, c[0]) as f32,
//             lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, c[1]) as f32,
//         );
//         draw_line_segment_mut(&mut image, a.clone(), b.clone(), red);
//         draw_line_segment_mut(&mut image, b.clone(), c.clone(), red);
//         draw_line_segment_mut(&mut image, c.clone(), a.clone(), red);
//     }
//
//     for [aidx, bidx, cidx] in triangles {
//         let a = vertices[*aidx];
//         let b = vertices[*bidx];
//         let c = vertices[*cidx];
//         let a = (
//             lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, a[0]) as f32,
//             lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, a[1]) as f32,
//         );
//         let b = (
//             lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, b[0]) as f32,
//             lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, b[1]) as f32,
//         );
//         let c = (
//             lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, c[0]) as f32,
//             lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, c[1]) as f32,
//         );
//         let ai = (a.0.round() as i32, a.1.round() as i32);
//         let bi = (b.0.round() as i32, b.1.round() as i32);
//         let ci = (c.0.round() as i32, c.1.round() as i32);
//         draw_filled_circle_mut(&mut image, ai, 5, light_blue);
//         draw_filled_circle_mut(&mut image, bi, 5, light_blue);
//         draw_filled_circle_mut(&mut image, ci, 5, light_blue);
//     }
//
//     for [aidx, bidx, cidx] in triangles {
//         let a = vertices[*aidx];
//         let b = vertices[*bidx];
//         let c = vertices[*cidx];
//         let a = (
//             lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, a[0]) as f32,
//             lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, a[1]) as f32,
//         );
//         let b = (
//             lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, b[0]) as f32,
//             lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, b[1]) as f32,
//         );
//         let c = (
//             lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, c[0]) as f32,
//             lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, c[1]) as f32,
//         );
//         let ai = (a.0.round() as i32, a.1.round() as i32);
//         let bi = (b.0.round() as i32, b.1.round() as i32);
//         let ci = (c.0.round() as i32, c.1.round() as i32);
//         let at = format!("{}", *aidx);
//         let bt = format!("{}", *bidx);
//         let ct = format!("{}", *cidx);
//         draw_text_mut(&mut image, blue, ai.0, ai.1, scale, &font, &at.as_str());
//         draw_text_mut(&mut image, blue, bi.0, bi.1, scale, &font, &bt.as_str());
//         draw_text_mut(&mut image, blue, ci.0, ci.1, scale, &font, &ct.as_str());
//     }
//
//     for (i, [aidx, bidx, cidx]) in triangles.iter().enumerate() {
//         let a = vertices[*aidx];
//         let b = vertices[*bidx];
//         let c = vertices[*cidx];
//         let centroid = [(a[0] + b[0] + c[0]) / 3., (a[1] + b[1] + c[1]) / 3.];
//         let cen = (
//             lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, centroid[0]) as f32,
//             lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, centroid[1]) as f32,
//         );
//         let ci = (cen.0.round() as i32, cen.1.round() as i32);
//         let ct = format!("{}", i);
//         draw_text_mut(&mut image, red, ci.0, ci.1, scale, &font, &ct.as_str());
//     }
//
//     for (i, [aidx, bidx, cidx]) in triangles.iter().enumerate() {
//         let a = vertices[*aidx];
//         let b = vertices[*bidx];
//         let c = vertices[*cidx];
//         let centroid = [(a[0] + b[0] + c[0]) / 3., (a[1] + b[1] + c[1]) / 3.];
//         let cen = (
//             lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, centroid[0]) as f32,
//             lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, centroid[1]) as f32,
//         );
//         let ci = (cen.0.round() as i32, cen.1.round() as i32);
//         draw_filled_circle_mut(&mut image, ci, 5, light_red);
//     }
//
//     if let Some(q) = q {
//         for e in q.0 {
//             let a = vertices[triangulation.triangles[*e]];
//             let b = vertices[triangulation.triangles[next_halfedge(*e)]];
//             let a = (
//                 lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, a[0]) as f32,
//                 lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, a[1]) as f32,
//             );
//             let b = (
//                 lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, b[0]) as f32,
//                 lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, b[1]) as f32,
//             );
//             draw_line_segment_mut(&mut image, a.clone(), b.clone(), green);
//         }
//
//         let e = &q.0[q.0.len() - 1];
//         let a = vertices[triangulation.triangles[*e]];
//         let b = vertices[triangulation.triangles[next_halfedge(*e)]];
//         let a = (
//             lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, a[0]) as f32,
//             lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, a[1]) as f32,
//         );
//         let b = (
//             lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, b[0]) as f32,
//             lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, b[1]) as f32,
//         );
//         draw_line_segment_mut(&mut image, a.clone(), b.clone(), purple);
//
//         let cen = (
//             lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, q.1[0]) as f32,
//             lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, q.1[1]) as f32,
//         );
//         let ci = (cen.0.round() as i32, cen.1.round() as i32);
//         draw_filled_circle_mut(&mut image, ci, 5, purple);
//     }
//
//     image.save(path).unwrap();
// }
