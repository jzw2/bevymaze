use crate::maze_render::{
    distance_to_arc, distance_to_circle, distance_to_segment, Circle, CircleArc, Segment,
};
use crate::terrain_render::{X_VIEW_DIST_M, Z_VIEW_DIST_M};
use ab_glyph::{FontRef, PxScale};
use bevy::math::DVec2;
use delaunator::{next_halfedge, Triangulation};
use image::{Rgb, RgbImage, Rgba, RgbaImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_line_segment_mut, draw_text_mut};
use itertools::enumerate;
use rand::prelude::StdRng;
use rand::{thread_rng, Rng, SeedableRng};
use server::util::lin_map;
use std::cmp::{max, min};
use svg::node::element;
use svg::node::element::path::Data;
use svg::node::element::tag::Circle;
use svg::node::element::Path;
use svg::Document;

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
    q: Option<(&Vec<[usize; 2]>, &[f64; 2])>,
    triangulation: &Triangulation,
    vertices: &Vec<[f64; 2]>,
    triangles: &Vec<[usize; 3]>,
    bounds: (f64, f64, f64, f64),
) {
    let mut document = Document::new().set("viewBox", bounds);

    // draw the triangles of the triangulation
    for [aidx, bidx, cidx] in triangles {
        let a = vertices[*aidx];
        let b = vertices[*bidx];
        let c = vertices[*cidx];

        let data = Data::new()
            .move_to((a[0], a[1]))
            .line_to((b[0], b[1]))
            .line_to((c[0], c[1]))
            .close();

        let path = Path::new()
            .set("fill", "none")
            .set("stroke", "red")
            .set("stroke-width", 0.1)
            .set("d", data);

        document = document.add(path);
    }

    // draw the vertices of the triangles
    for [aidx, bidx, cidx] in triangles {
        let a = vertices[*aidx];
        let b = vertices[*bidx];
        let c = vertices[*cidx];

        let ca = element::Circle::new()
            .set("cx", a[0])
            .set("cy", a[1])
            .set("r", 0.1);
        let cb = element::Circle::new()
            .set("cx", b[0])
            .set("cy", b[1])
            .set("r", 0.1);
        let cc = element::Circle::new()
            .set("cx", c[0])
            .set("cy", c[1])
            .set("r", 0.1);

        document = document.add(ca).add(cb).add(cc);
    }

    // draw the index of the vertices of the triangulation
    for [aidx, bidx, cidx] in triangles {
        let a = vertices[*aidx];
        let b = vertices[*bidx];
        let c = vertices[*cidx];

        let at = format!("{}", *aidx);
        let bt = format!("{}", *bidx);
        let ct = format!("{}", *cidx);

        let ca = element::Text::new(at)
            .set("x", a[0])
            .set("y", a[1])
            .set("fill", "blue")
            .set("font-size", "0.2");
        let cb = element::Text::new(bt)
            .set("x", b[0])
            .set("y", b[1])
            .set("fill", "blue")
            .set("font-size", "0.2");
        let cc = element::Text::new(ct)
            .set("x", c[0])
            .set("y", c[1])
            .set("fill", "blue")
            .set("font-size", "0.2");

        document = document.add(ca).add(cb).add(cc);
    }

    // draw the index of the triangles of the triangulation and the centroids
    for (i, [aidx, bidx, cidx]) in triangles.iter().enumerate() {
        let a = vertices[*aidx];
        let b = vertices[*bidx];
        let c = vertices[*cidx];
        let centroid = [(a[0] + b[0] + c[0]) / 3., (a[1] + b[1] + c[1]) / 3.];

        let centc = element::Circle::new()
            .set("cx", centroid[0])
            .set("cy", centroid[1])
            .set("r", 0.05);

        document = document.add(centc);
    }

    // draw the index of the triangles of the triangulation and the centroids
    for (i, [aidx, bidx, cidx]) in triangles.iter().enumerate() {
        let a = vertices[*aidx];
        let b = vertices[*bidx];
        let c = vertices[*cidx];
        let centroid = [(a[0] + b[0] + c[0]) / 3., (a[1] + b[1] + c[1]) / 3.];

        let ct = format!("{}", i);

        let cc = element::Text::new(ct)
            .set("x", centroid[0])
            .set("y", centroid[1])
            .set("fill", "blue")
            .set("font-size", "0.2");

        document = document.add(cc);
    }

    // draw the path taken through the triangulation
    if let Some(q) = q {
        // draw the edges taken
        for (i, edge) in q.0.iter().enumerate() {
            let [aidx, bidx] = edge;
            let a = vertices[triangulation.triangles[*aidx]];
            let b = vertices[triangulation.triangles[*bidx]];
            let mut r = StdRng::seed_from_u64(*aidx as u64);

            let color = format!(
                "rgb({},{},{})",
                r.gen_range(0..255),
                r.gen_range(0..255),
                r.gen_range(0..255)
            );
            let data = Data::new().move_to((a[0], a[1])).line_to((b[0], b[1]));

            let path = Path::new()
                .set("fill", "none")
                .set("stroke", color)
                .set("stroke-width", 0.1)
                .set("d", data);

            document = document.add(path);
        }
        // draw the label for the edges
        for (i, edge) in q.0.iter().enumerate() {
            let [aidx, bidx] = edge;
            let a = vertices[triangulation.triangles[*aidx]];
            let b = vertices[triangulation.triangles[*bidx]];

            let ct = format!("{}", i);

            let cc = element::Text::new(ct)
                .set("x", (a[0] + b[0]) / 2.0)
                .set("y", (a[1] + b[1]) / 2.0)
                .set("fill", "green")
                .set("font-size", "0.2");

            document = document.add(cc);
        }
    }

    svg::save(path, &document).unwrap();
}
