use crate::maze_render::{
    distance_to_arc, distance_to_circle, distance_to_segment, Arc, Circle, Segment,
};
use image::{Rgb, RgbImage};
use std::cmp::{max, min};

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
    pub(crate) arc: Arc,
    pub(crate) color: Rgb<u8>,
    pub(crate) line_width: f64,
}

#[derive(Clone, Copy)]
pub struct AxisTransform {
    pub(crate) offset: (f64, f64),
    pub(crate) scale: (f64, f64),
}

pub fn to_canvas_space(pixel_space_coordinate: (u32, u32), transform: AxisTransform) -> (f64, f64) {
    return (
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
