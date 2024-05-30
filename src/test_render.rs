use crate::maze_render::{
    distance_to_arc, distance_to_circle, distance_to_segment, Circle, CircleArc, Segment,
};
use crate::terrain_render::{X_VIEW_DIST_M, Z_VIEW_DIST_M};
use ab_glyph::{FontRef, PxScale};
use bevy::math::DVec2;
use delaunator::{next_halfedge, Triangulation};
use image::{Rgb, RgbImage, Rgba, RgbaImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_line_segment_mut, draw_text_mut};
use server::util::lin_map;
use std::cmp::{max, min};
use svg::Document;
use svg::node::element::Path;
use svg::node::element::path::Data;

pub struct DrawableCircle {
    pub(crate) circle: Circle,
    pub(crate) line_width: f64,
    pub(crate) color: Rgb<u8>,
}

pub struct DrawableSegment {
    pub(crate) segment: Segment,
    pub(crate) color: Rgb<u8>,
    pub(crate) line_width: f64,
}

pub struct DrawableArc {
    pub(crate) arc: CircleArc,
    pub(crate) color: Rgb<u8>,
    pub(crate) line_width: f64,
}

#[derive(Clone, Copy)]
pub struct AxisTransform {
    pub(crate) offset: (f64, f64),
    pub(crate) scale: (f64, f64),
}

pub fn to_canvas_space(pixel_space_coordinate: (u32, u32), transform: AxisTransform) -> DVec2 {
    return DVec2::new(
        (pixel_space_coordinate.0 as f64 - transform.offset.0) / transform.scale.0,
        (pixel_space_coordinate.1 as f64 - transform.offset.1) / transform.scale.1,
    );
}

pub fn draw_circle(image: &mut RgbImage, circle: DrawableCircle, transform: AxisTransform) {
    for px in 0..image.width() {
        for py in 0..image.height() {
            let circle_dist =
                distance_to_circle(&circle.circle, to_canvas_space((px, py), transform));
            if circle_dist <= circle.line_width / 2.0 {
                image.put_pixel(px, py, circle.color);
            }
        }
    }
}

pub fn draw_segment(image: &mut RgbImage, segment: DrawableSegment, transform: AxisTransform) {
    for px in 0..image.width() {
        for py in 0..image.height() {
            let seg_dist =
                distance_to_segment(&segment.segment, to_canvas_space((px, py), transform));
            if seg_dist <= segment.line_width / 2.0 {
                image.put_pixel(px, py, segment.color);
            }
        }
    }
}

pub fn draw_arc(image: &mut RgbImage, arc: DrawableArc, transform: AxisTransform) {
    for px in 0..image.width() {
        for py in 0..image.height() {
            let seg_dist = distance_to_arc(&arc.arc, to_canvas_space((px, py), transform));
            if seg_dist <= arc.line_width / 2.0 {
                image.put_pixel(px, py, arc.color);
            }
        }
    }
}

pub fn draw_debug(
    path: &str,
    q: Option<(&Vec<usize>, &[f64; 2])>,
    triangulation: &Triangulation,
    vertices: &Vec<[f64; 2]>,
    triangles: &Vec<[usize; 3]>,
) {
    // Background color
    let white = Rgba([255u8, 255u8, 255u8, 255u8]);

    // Drawing color
    let red = Rgba([255u8, 0u8, 0u8, 255u8]);
    let light_red = Rgba([255u8, 150u8, 150u8, 255u8]);
    let blue = Rgba([0u8, 0u8, 255u8, 255u8]);
    let light_blue = Rgba([150u8, 150u8, 255u8, 255u8]);
    let green = Rgba([0u8, 255u8, 0u8, 255u8]);
    let purple = Rgba([255u8, 0u8, 255u8, 255u8]);

    let dim = 1024;

    // The implementation of Canvas for GenericImage overwrites existing pixels
    let mut image = RgbaImage::from_pixel(dim, dim, white);

    let font = FontRef::try_from_slice(include_bytes!(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ))
    .unwrap();

    let height = 24.;

    let scale = PxScale {
        x: height,
        y: height,
    };

    let mut document = Document::new()
        .set("viewBox", (-X_VIEW_DIST_M, -Z_VIEW_DIST_M, 2.*X_VIEW_DIST_M, 2.*Z_VIEW_DIST_M));
    
    // draw the triangles of the triangulation
    for [aidx, bidx, cidx] in triangles {
        let a = vertices[*aidx];
        let b = vertices[*bidx];
        let c = vertices[*cidx];

        let data = Data::new()
            .move_to((a[0], a[1]))
            .line_by((b[0], b[1]))
            .line_by((c[0], c[1]))
            .close();
        
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
        draw_line_segment_mut(&mut image, a.clone(), b.clone(), red);
        draw_line_segment_mut(&mut image, b.clone(), c.clone(), red);
        draw_line_segment_mut(&mut image, c.clone(), a.clone(), red);

        let path = Path::new()
            .set("fill", "none")
            .set("stroke", "black")
            .set("stroke-width", 3)
            .set("d", data);

        document = document.add(path);
    }

    // draw the vertices of the triangles
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
        draw_filled_circle_mut(&mut image, ai, 5, light_blue);
        draw_filled_circle_mut(&mut image, bi, 5, light_blue);
        draw_filled_circle_mut(&mut image, ci, 5, light_blue);
    }

    // draw the index of the vertices of the triangulation
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

    // draw the index of the triangles of the triangulation
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

    // draw the centroids
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

    // draw the path taken through the triangulation
    if let Some(q) = q {
        for e in q.0 {
            let a = vertices[triangulation.triangles[*e]];
            let b = vertices[triangulation.triangles[next_halfedge(*e)]];
            let a = (
                lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, a[0]) as f32,
                lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, a[1]) as f32,
            );
            let b = (
                lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, b[0]) as f32,
                lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, b[1]) as f32,
            );
            draw_line_segment_mut(&mut image, a.clone(), b.clone(), green);
        }

        let e = &q.0[q.0.len() - 1];
        let a = vertices[triangulation.triangles[*e]];
        let b = vertices[triangulation.triangles[next_halfedge(*e)]];
        let a = (
            lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, a[0]) as f32,
            lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, a[1]) as f32,
        );
        let b = (
            lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, b[0]) as f32,
            lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, b[1]) as f32,
        );
        draw_line_segment_mut(&mut image, a.clone(), b.clone(), purple);

        let cen = (
            lin_map(-X_VIEW_DIST_M, X_VIEW_DIST_M, 0., dim as f64, q.1[0]) as f32,
            lin_map(-Z_VIEW_DIST_M, Z_VIEW_DIST_M, 0., dim as f64, q.1[1]) as f32,
        );
        let ci = (cen.0.round() as i32, cen.1.round() as i32);
        draw_filled_circle_mut(&mut image, ci, 5, purple);
    }

    image.save(path).unwrap();
    svg::save("image.svg", &document).unwrap();
    
}
